# CacheBench v1 — Results for `CacheInference (Ours)`

## Headline (binary HIT/MISS classification)

- **Precision:** 0.3155
- **Recall:** 0.2533
- **F1:** 0.2810
- **False-Hit Rate (FHR):** 0.2800  ← safety-critical
- **Accuracy:** 0.5625  (95% Wilson CI: 0.541, 0.584)
- **N:** 2000

## Confusion matrix (overall)

| | predicted HIT | predicted MISS |
|---|---|---|
| truth HIT  | TP=171 | FN=504 |
| truth MISS | FP=371 | TN=954 |

## Per-domain breakdown

| Domain | N | F1 | FHR | Precision | Recall |
|---|---|---|---|---|---|
| code | 160 | 0.473 | **0.110** | 0.667 | 0.367 |
| conversational | 260 | 0.089 | **0.364** | 0.136 | 0.067 |
| creative | 180 | 0.000 | **0.439** | 0.000 | 0.000 |
| math | 160 | 0.391 | **0.140** | 0.562 | 0.300 |
| multi_turn | 180 | 0.052 | **0.433** | 0.055 | 0.050 |
| multilingual | 150 | 0.062 | **0.130** | 0.133 | 0.040 |
| personalized | 140 | 0.000 | **0.629** | 0.000 | 0.000 |
| qa_factual | 230 | 0.237 | **0.031** | 0.778 | 0.140 |
| qa_open | 260 | 0.796 | **0.029** | 0.953 | 0.683 |
| tool | 280 | 0.242 | **0.314** | 0.286 | 0.210 |

## Per-label FHR (false-hit rate by label class — the safety dimensions)

| Label | N | FP | FHR |
|---|---|---|---|
| RELATED_UNSAFE | 420 | 157 | 0.374 |
| ADVERSARIAL | 410 | 107 | 0.261 |
| UNRELATED | 340 | 2 | 0.006 |

## Per-difficulty breakdown

| Difficulty | N | F1 | FHR |
|---|---|---|---|
| easy | 1184 | 0.399 | 0.117 |
| medium | 238 | 0.206 | 0.261 |
| hard | 578 | 0.169 | 0.678 |

## Latency

- p50: 0.01 ms
- p95: 0.02 ms
- p99: 0.03 ms
- mean: 0.01 ms

## Cost

- total: $0.0000
- per 1k decisions: $0.0000

## Verification-tier usage (which check made the call?)

| Tier | Count | % |
|---|---|---|
| embedding | 1258 | 62.9% |
| verify | 500 | 25.0% |
| exact | 242 | 12.1% |
