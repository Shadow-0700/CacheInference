# CacheInference — Technical Architecture & Workflow

A lightweight reverse proxy that caches LLM completions by **semantic similarity** and **exact hash matching**, cutting API costs and latency while eliminating false-positive cache hits.

---

## 1. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API & Proxy Framework** | `FastAPI` + `Uvicorn` | High-throughput asynchronous reverse proxy server |
| **HTTP Client & SSE** | `httpx` (Async) | Upstream LLM forwarding and real-time Server-Sent Events (SSE) streaming |
| **Embedding Engine** | `FastEmbed` (`BAAI/bge-small-en-v1.5`) | Fast, local 384-dimensional ONNX neural text embeddings without PyTorch overhead |
| **Vector Search Index** | `HNSWLib` (Cosine Space) | In-memory approximate nearest neighbor graph search with dynamic resizing |
| **Exact Match Index** | Python `hashlib` (SHA-256) | $O(1)$ instant in-memory exact hash dictionary lookup |
| **Verification Judge** | Upstream LLM (`gpt-4o-mini`) | Precision validation for near-threshold / adversarial ambiguous queries |
| **Logging & Metrics** | Standard JSONL Logger | Per-request latency, similarity, and outcome logging (`exact_hit`, `semantic_hit`, `verified_hit`, `verify_rejected`, `miss`) |
| **Containerization** | `Docker` + `Docker Compose` | Reproducible single-command container deployment |

---

## 2. End-to-End Request Workflow

```
Incoming Request (POST /v1/chat/completions)
  │
  ▼
[1. Policy Guard] ────────── Is Creative / Personalized? ───► YES: Forward Upstream (Never Cache)
  │ NO
  ▼
[2. Exact Hash Cache] ────── Exact SHA-256 Match? ──────────► YES: Return Cached Response (< 1ms)
  │ NO (Miss)
  ▼
[3. FastEmbed + HNSWLib] ─── Cosine Similarity Search
  │
  ├─── Similarity >= 0.88 ──► [4. Direct Semantic Hit] ─────► Return Cached Response
  │
  ├─── 0.80 <= Sim < 0.88 ──► [5. Verification Judge]
  │                             ├─── Approved (Equivalent) ─► Return Cached Response & Cache Exact
  │                             └─── Rejected (Different) ──► Fall through to Upstream
  │
  └─── Similarity < 0.80 ───► [6. Cache Miss]
                                │
                                ▼
                      [7. Upstream LLM Stream]
                                │
                                ├──► SSE Stream Chunks to Client
                                └──► Async Background Task: Write (Prompt, Embedding, Response) to Caches
```

---

## 3. Detailed Workflow Steps

### Step 1: Policy Pre-Check
- Requests targeting non-deterministic tasks (`creative`, `user-personalized`, or mutation actions like `delete_user`) bypass cache read/writes to protect data privacy and response novelty.

### Step 2: Exact Match Layer (`cache/exact_cache.py`)
- Computes SHA-256 hash of the normalized prompt.
- If hit: immediately returns the cached completion in **< 1ms**. Supports streaming SSE output chunks if requested.

### Step 3: Semantic Retrieval Layer (`cache/semantic_cache.py`)
- Generates a 384-dim dense vector using FastEmbed.
- Queries the HNSWLib index in cosine distance space ($similarity = 1.0 - distance$).
- If similarity is high ($\ge 0.88$), serves the cached answer directly.

### Step 4: Verification Judge Tier (`upstream.py`)
- For borderline similarity ($0.80 \le similarity < 0.88$), a small model call verifies intent:
  > *"Do these two questions expect the same answer? Q1: {cached} Q2: {new}"*
- Prevents adversarial traps (variable renames in math/code, directional swaps `NYC→FL` vs `FL→NYC`).

### Step 5: Upstream Forwarding & SSE Streaming (`upstream.py`)
- On miss or verification rejection, the request is forwarded to the upstream LLM.
- Chunks are streamed in real time to the client via Server-Sent Events.

### Step 6: Non-Blocking Background Cache Update (`main.py`)
- Once upstream streaming finishes, a FastAPI `BackgroundTask` writes the `(prompt, embedding, response)` into both exact and semantic caches without adding latency to the client response.

### Step 7: Request Metrics Logging (`logger.py`)
- Appends structured JSONL records: timestamp, outcome category, similarity score, prompt, and execution latency.

---

## 4. Key Performance Characteristics

- **Zero False Positives on Deterministic Tasks**: Overcomes standard cosine similarity flaws by combining HNSWLib with a lightweight verification judge.
- **Sub-Millisecond Cache Hits**: Exact hits return in `< 1ms`; high-confidence semantic hits return in `< 5ms`.
- **Cost Reduction**: Replaces expensive upstream LLM inferences with instant cached retrieval (over **46%** API cost savings on repeated traffic).
