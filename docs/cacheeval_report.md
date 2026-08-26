# CacheBench v1 — Results for `CacheInference (High-Precision)`

## Headline (binary HIT/MISS classification)

- **Precision:** 0.8023
- **Recall:** 0.9259
- **F1:** 0.8597
- **False-Hit Rate (FHR):** 0.1162  ← safety-critical
- **Accuracy:** 0.8980  (95% Wilson CI: 0.884, 0.911)
- **N:** 2000

## Confusion matrix (overall)

| | predicted HIT | predicted MISS |
|---|---|---|
| truth HIT  | TP=625 | FN=50 |
| truth MISS | FP=154 | TN=1171 |

## Per-domain breakdown

| Domain | N | F1 | FHR | Precision | Recall |
|---|---|---|---|---|---|
| code | 160 | 0.984 | **0.020** | 0.968 | 1.000 |
| conversational | 260 | 0.804 | **0.364** | 0.693 | 0.958 |
| creative | 180 | 0.000 | **0.000** | 0.000 | 0.000 |
| math | 160 | 0.941 | **0.030** | 0.949 | 0.933 |
| multi_turn | 180 | 0.686 | **0.425** | 0.532 | 0.967 |
| multilingual | 150 | 0.723 | **0.030** | 0.909 | 0.600 |
| personalized | 140 | 0.000 | **0.000** | 0.000 | 0.000 |
| qa_factual | 230 | 0.965 | **0.023** | 0.970 | 0.960 |
| qa_open | 260 | 0.949 | **0.029** | 0.966 | 0.933 |
| tool | 280 | 0.817 | **0.211** | 0.726 | 0.933 |

## Per-label FHR (false-hit rate by label class — the safety dimensions)

| Label | N | FP | FHR |
|---|---|---|---|
| RELATED_UNSAFE | 420 | 120 | 0.286 |
| ADVERSARIAL | 410 | 19 | 0.046 |
| UNRELATED | 340 | 0 | 0.000 |

## Per-difficulty breakdown

| Difficulty | N | F1 | FHR |
|---|---|---|---|
| easy | 1184 | 0.893 | 0.066 |
| medium | 238 | 0.913 | 0.083 |
| hard | 578 | 0.795 | 0.252 |

## Latency

- p50: 24.75 ms
- p95: 36.72 ms
- p99: 37.79 ms
- mean: 18.20 ms

## Cost

- total: $0.0426
- per 1k decisions: $0.0213

## Verification-tier usage (which check made the call?)

| Tier | Count | % |
|---|---|---|
| verify | 1217 | 60.9% |
| embedding | 341 | 17.1% |
| policy | 320 | 16.0% |
| exact | 122 | 6.1% |
