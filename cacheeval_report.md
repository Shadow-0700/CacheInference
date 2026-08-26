# CacheBench v1 — Results for `CacheInference (LLM-Verified)`

## Headline (binary HIT/MISS classification)

- **Precision:** 0.6536
- **Recall:** 0.8859
- **F1:** 0.7522
- **False-Hit Rate (FHR):** 0.2392  ← safety-critical
- **Accuracy:** 0.8030  (95% Wilson CI: 0.785, 0.820)
- **N:** 2000

## Confusion matrix (overall)

| | predicted HIT | predicted MISS |
|---|---|---|
| truth HIT  | TP=598 | FN=77 |
| truth MISS | FP=317 | TN=1008 |

## Per-domain breakdown

| Domain | N | F1 | FHR | Precision | Recall |
|---|---|---|---|---|---|
| code | 160 | 0.918 | **0.060** | 0.903 | 0.933 |
| conversational | 260 | 0.787 | **0.364** | 0.685 | 0.925 |
| creative | 180 | 0.000 | **0.244** | 0.000 | 0.000 |
| math | 160 | 0.959 | **0.030** | 0.951 | 0.967 |
| multi_turn | 180 | 0.644 | **0.483** | 0.491 | 0.933 |
| multilingual | 150 | 0.459 | **0.070** | 0.708 | 0.340 |
| personalized | 140 | 0.000 | **0.579** | 0.000 | 0.000 |
| qa_factual | 230 | 0.895 | **0.038** | 0.944 | 0.850 |
| qa_open | 260 | 0.963 | **0.043** | 0.951 | 0.975 |
| tool | 280 | 0.757 | **0.320** | 0.636 | 0.933 |

## Per-label FHR (false-hit rate by label class — the safety dimensions)

| Label | N | FP | FHR |
|---|---|---|---|
| RELATED_UNSAFE | 420 | 148 | 0.352 |
| ADVERSARIAL | 410 | 79 | 0.193 |
| UNRELATED | 340 | 1 | 0.003 |

## Per-difficulty breakdown

| Difficulty | N | F1 | FHR |
|---|---|---|---|
| easy | 1184 | 0.863 | 0.115 |
| medium | 238 | 0.826 | 0.172 |
| hard | 578 | 0.583 | 0.565 |

## Latency

- p50: 24.17 ms
- p95: 36.56 ms
- p99: 37.76 ms
- mean: 17.57 ms

## Cost

- total: $0.0411
- per 1k decisions: $0.0205

## Verification-tier usage (which check made the call?)

| Tier | Count | % |
|---|---|---|
| verify | 1173 | 58.6% |
| embedding | 585 | 29.2% |
| exact | 242 | 12.1% |
