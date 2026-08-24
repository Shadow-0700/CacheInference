# CacheInference — Build Spec

For a coding agent. Build in the order below. Keep it simple — this is an intern-level proof of concept, not production infra.

---

## Goal

A FastAPI reverse proxy that caches LLM completions by semantic similarity, not just exact match, to cut latency and cost on repeated/similar prompts.

---

## Project structure

```
cacheinference/
├── main.py                # FastAPI app, proxy endpoint
├── config.py              # env vars: API key, thresholds
├── cache/
│   ├── exact_cache.py     # hash-based dict cache
│   └── semantic_cache.py  # fastembed + hnswlib wrapper
├── upstream.py            # forwards request to upstream LLM, handles SSE
├── logger.py              # logs hit/verify/miss + latency + similarity
├── benchmark.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Build order

### 1. Exact cache
- `exact_cache.py`: dict keyed by SHA256 hash of the prompt string.
- `get(prompt) -> response | None`, `set(prompt, response)`.

### 2. Proxy skeleton
- `main.py`: single POST endpoint `/v1/chat/completions` (mimic OpenAI-style schema).
- On request: check exact cache first. If hit, return immediately.

### 3. Semantic cache
- `semantic_cache.py`: on miss, embed prompt with `fastembed` (`BAAI/bge-small-en-v1.5`, 384-dim).
- Query `hnswlib` index (cosine space) for nearest neighbor + similarity score.
- Three outcomes:
  - `similarity > 0.97` → return cached answer directly
  - `0.90 <= similarity <= 0.97` → mark as "needs verification" (see step 5)
  - `similarity < 0.90` → treat as full miss

### 4. Upstream forwarding
- `upstream.py`: on full miss, call the upstream LLM API, stream response back to client via SSE.
- After stream completes, write `(prompt, embedding, response)` into both caches in a background task — do not block the client response on this.

### 5. Verification for near-threshold hits
- Simple version: ask the same upstream LLM a short yes/no prompt — "Do these two questions expect the same answer? Q1: {cached_prompt} Q2: {new_prompt}"
- If yes → serve cached answer. If no → treat as full miss, forward upstream.
- Keep this as a single small LLM call, not a trained model, for v1.

### 6. Logging
- `logger.py`: append one line per request — timestamp, outcome (exact_hit / semantic_hit / verified_hit / verify_rejected / miss), similarity score, latency_ms.
- Plain CSV or JSONL, no DB needed.

### 7. Benchmark script
- Separate script `benchmark.py`: replay a list of prompts (mix of exact repeats, paraphrases, and unrelated questions) through the proxy.
- Compare against a no-cache baseline run.
- Output: hit rate, false-positive rate, avg latency, estimated cost saved. Print as a table.

### 8. GPTCache comparison
- Install `gptcache` separately, set up its default semantic caching on the same dataset used in step 7.
- Run the identical prompt set through both CacheInference and GPTCache.
- Log hit rate, false-positive rate, and latency for both. Add both to the same results table.

### 9. Dockerize
- `Dockerfile`: simple Python base image, install `requirements.txt`, run `uvicorn main:app`.
- `docker-compose.yml`: one service for the app, exposing the port, loading `.env`.
- Should run with a single `docker compose up` — no other setup steps.

---

## Config (.env)

```
UPSTREAM_API_KEY=
UPSTREAM_URL=
SIMILARITY_THRESHOLD=0.97
VERIFY_THRESHOLD=0.90
```

---

## Explicit non-goals (do not build these)

- No distributed/multi-node index
- No persistence/snapshotting of the index across restarts (v1 is in-memory only)
- No multi-tenant/API-key isolation
- No auth layer
- No trained classifier for verification — use a single LLM call
