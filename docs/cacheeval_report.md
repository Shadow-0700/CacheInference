# CacheBench v1 — Results for `CacheInference (Ours)`

## Headline (binary HIT/MISS classification)

- **Precision:** 0.5100
- **Recall:** 0.3643
- **F1:** 0.4250
- **False-Hit Rate (FHR):** 0.8167  ← safety-critical
- **Accuracy:** 0.3100  (95% Wilson CI: 0.250, 0.377)
- **N:** 200

## Confusion matrix (overall)

| | predicted HIT | predicted MISS |
|---|---|---|
| truth HIT  | TP=51 | FN=89 |
| truth MISS | FP=49 | TN=11 |

## Per-domain breakdown

| Domain | N | F1 | FHR | Precision | Recall |
|---|---|---|---|---|---|
| code | 20 | 0.710 | **0.000** | 1.000 | 0.550 |
| conversational | 20 | 0.000 | **0.000** | 0.000 | 0.000 |
| creative | 20 | 0.000 | **1.000** | 0.000 | 0.000 |
| math | 20 | 0.889 | **0.000** | 1.000 | 0.800 |
| multi_turn | 20 | 0.182 | **0.000** | 1.000 | 0.100 |
| multilingual | 20 | 0.000 | **0.450** | 0.000 | 0.000 |
| personalized | 20 | 0.000 | **1.000** | 0.000 | 0.000 |
| qa_factual | 20 | 0.400 | **0.000** | 1.000 | 0.250 |
| qa_open | 20 | 0.919 | **0.000** | 1.000 | 0.850 |
| tool | 20 | 0.000 | **0.000** | 0.000 | 0.000 |

## Per-label FHR (false-hit rate by label class — the safety dimensions)

| Label | N | FP | FHR |
|---|---|---|---|
| RELATED_UNSAFE | 0 | 0 | 0.000 |
| ADVERSARIAL | 20 | 9 | 0.450 |
| UNRELATED | 0 | 0 | 0.000 |

## Per-difficulty breakdown

| Difficulty | N | F1 | FHR |
|---|---|---|---|
| easy | 138 | 0.560 | 0.000 |
| medium | 3 | 0.000 | 0.000 |
| hard | 59 | 0.000 | 0.961 |

## Latency

- p50: 0.01 ms
- p95: 0.04 ms
- p99: 0.06 ms
- mean: 0.02 ms

## Cost

- total: $0.0000
- per 1k decisions: $0.0000

## Verification-tier usage (which check made the call?)

| Tier | Count | % |
|---|---|---|
| embedding | 88 | 44.0% |
| verify | 72 | 36.0% |
| exact | 40 | 20.0% |
