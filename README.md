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

## Benchmark Results

Evaluated over a mixed dataset (exact repeats, paraphrases, and negatives):

| System | Total Requests | Hit Rate (%) | False Positive Rate (%) | Avg Latency (ms) | Est. Cost Saved (%) |
|---|---|---|---|---|---|
| No-Cache Baseline | 13 | 0.0% | 0.0% | 350.40 ms | 0.0% |
| **CacheInference (Ours)** | 13 | **30.8%** | **0.0%** | **294.62 ms** | **30.8%** |
| GPTCache (Standard) | 13 | 38.5% | 7.7% | 228.40 ms | 38.5% |

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

