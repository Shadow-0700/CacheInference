# CacheInference

A lightweight FastAPI reverse proxy that caches LLM completions using semantic similarity and exact hash matching to reduce latency and API costs.

---

## Architecture Overview

```
Client Request
      │
      ▼
┌──────────────────┐
│  Exact Cache     │─── Hit (SHA256 Match) ───► Return Cached Response
└─────────┬────────┘
          │ Miss
          ▼
┌──────────────────┐
│  Semantic Cache  │─── Similarity > 0.97 ────► Return Cached Response
└─────────┬────────┘
          │ 0.90 <= Similarity <= 0.97
          ▼
┌──────────────────┐
│ LLM Verification │─── Yes (Equivalent) ─────► Return Cached Response
└─────────┬────────┘
          │ No / Similarity < 0.90 (Miss)
          ▼
┌──────────────────┐
│   Upstream LLM   │─── SSE Stream Response ──► Client & Update Caches Asynchronously
└──────────────────┘
```

---

## Features

- **Exact Cache (`cache/exact_cache.py`)**: Instant in-memory SHA256 prompt matching.
- **Semantic Cache (`cache/semantic_cache.py`)**: Vector similarity search powered by `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim) and `hnswlib` cosine space.
- **Near-Threshold Verification (`upstream.py`)**: Smart yes/no verification via upstream LLM for queries in the `[0.90, 0.97]` similarity range to eliminate false positives.
- **Async Upstream Forwarding (`upstream.py`)**: Non-blocking SSE streaming and background cache updates.
- **Structured Request Logging (`logger.py`)**: Per-request metrics logged to JSONL (`timestamp`, `outcome`, `similarity`, `latency_ms`).
- **Benchmark Suite (`benchmark.py`)**: Compares CacheInference against a no-cache baseline and standard GPTCache.

---

## Benchmark Results (2,000-Row CacheEval Testbed)

Evaluated on the standardized **[BudEcosystem/CacheEval](https://github.com/BudEcosystem/CacheEval)** 2,000-pair dataset across 10 domains (*code, math, qa_factual, qa_open, conversational, tool, creative, personalized, multi_turn, multilingual*):

| Metric | CacheInference (Ours) | GPTCache (Standard) | Advantage / Impact |
|---|---|---|---|
| **Total Test Pairs** | 2,000 | 2,000 | Identical Ground Truth |
| **Overall Accuracy** | **90.0%** | 57.5% | **+32.5% higher** |
| **Precision** | **80.7%** | 43.0% | **+37.7% higher** |
| **Recall (Equivalent Hit Rate)** | **92.4%** | 79.7% | **+12.7% higher** |
| **False-Hit Rate (Safety)** | **11.2%** | 53.8% | **42.6% safer** (eliminates false hits) |
| **F1 Score** | **0.862** | 0.559 | **+0.303** |
| **Decision Latency (p50)** | 24.30 ms | 3.50 ms | GPTCache is faster but serves wrong answers |
| **Decision Cost (per 1k)** | $0.0213 | $0.0000 | < $0.03 per 1,000 queries |

> **Key Takeaway:** Standard vector caches like GPTCache suffer a **53.8% False-Hit Rate** on adversarial pairs (e.g. `x+2=8` vs `a+2=8` or directional swaps `NYC→FL` vs `FL→NYC`). CacheInference’s **Verification Judge Tier** brings overall accuracy to **90.0%** and true equivalent recall to **92.4%**. Full breakdown in [`docs/cacheeval_comparison_report.md`](docs/cacheeval_comparison_report.md).


---

## Quickstart

### 1. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your environment variables in `.env`:
```ini
UPSTREAM_API_KEY=your_openai_or_compatible_api_key
UPSTREAM_URL=https://api.openai.com/v1
SIMILARITY_THRESHOLD=0.97
VERIFY_THRESHOLD=0.90
VERIFY_MODEL=gpt-4o-mini
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

The proxy will be accessible at `http://localhost:8000`.

### 3. Run Locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage Example

Send standard OpenAI-compatible completions requests:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

---

## Project Structure

```
CacheInference/
├── cache/                  # Core in-memory caching engines
│   ├── exact_cache.py      # SHA-256 hash-based prompt cache
│   └── semantic_cache.py   # FastEmbed + HNSWLib cosine similarity cache
├── eval/                   # Benchmarking & evaluation suites
│   ├── eval_comparison_2000.py # Full 2,000-pair comparison runner
│   ├── benchmark.py        # Baseline vs CacheInference vs GPTCache
│   ├── eval_harness.py     # Evaluation metrics engine
│   └── cachebench.jsonl    # Ground truth 2,000-row dataset
├── docs/                   # Documentation & benchmark reports
│   ├── PROJECT_OVERVIEW.md # Architecture & technical workflow
│   ├── cacheeval_comparison_report.md # Head-to-head 2,000-row comparison
│   ├── cacheeval_report.md # Full 2,000-pair CacheEval domain breakdown
│   └── SPEC (1).md         # Original specification
├── docker/                 # Deployment definitions
│   ├── Dockerfile
│   └── docker-compose.yml
├── main.py                 # FastAPI reverse proxy (/v1/chat/completions)
├── config.py               # Environment variables and threshold loader
├── upstream.py             # SSE streaming forwarder & LLM verification judge
├── logger.py               # JSONL structured request logger
├── requirements.txt        # Project dependencies
├── .env.example            # Environment configuration template
└── README.md               # Main documentation
```

---

## Running Benchmarks

### 1. 2,000-Row CacheEval Head-to-Head Comparison (Ours vs GPTCache)
```bash
python eval/eval_comparison_2000.py
```
*(Results saved in `docs/cacheeval_comparison_report.md`)*

### 2. General Prompt Evaluation Suite
```bash
python eval/benchmark.py
```

