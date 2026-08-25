# CacheBench v1 — Results for `CacheInference (Enhanced)`

## Headline (binary HIT/MISS classification)

- **Precision:** 0.4487
- **Recall:** 0.7704
- **F1:** 0.5671
- **False-Hit Rate (FHR):** 0.4823  ← safety-critical
- **Accuracy:** 0.6030  (95% Wilson CI: 0.581, 0.624)
- **N:** 2000

## Confusion matrix (overall)

| | predicted HIT | predicted MISS |
|---|---|---|
| truth HIT  | TP=520 | FN=155 |
| truth MISS | FP=639 | TN=686 |

## Per-domain breakdown

| Domain | N | F1 | FHR | Precision | Recall |
|---|---|---|---|---|---|
| code | 160 | 0.702 | **0.490** | 0.546 | 0.983 |
| conversational | 260 | 0.728 | **0.514** | 0.604 | 0.917 |
| creative | 180 | 0.000 | **0.222** | 0.000 | 0.000 |
| math | 160 | 0.643 | **0.370** | 0.554 | 0.767 |
| multi_turn | 180 | 0.548 | **0.692** | 0.394 | 0.900 |
| multilingual | 150 | 0.295 | **0.250** | 0.342 | 0.260 |
| personalized | 140 | 0.000 | **0.571** | 0.000 | 0.000 |
| qa_factual | 230 | 0.630 | **0.469** | 0.548 | 0.740 |
| qa_open | 260 | 0.652 | **0.636** | 0.532 | 0.842 |
| tool | 280 | 0.465 | **0.589** | 0.380 | 0.600 |

## Per-label FHR (false-hit rate by label class — the safety dimensions)

| Label | N | FP | FHR |
|---|---|---|---|
| RELATED_UNSAFE | 420 | 263 | 0.626 |
| ADVERSARIAL | 410 | 291 | 0.710 |
| UNRELATED | 340 | 2 | 0.006 |

## Per-difficulty breakdown

| Difficulty | N | F1 | FHR |
|---|---|---|---|
| easy | 1184 | 0.632 | 0.356 |
| medium | 238 | 0.609 | 0.459 |
| hard | 578 | 0.459 | 0.794 |

## Latency

- p50: 0.01 ms
- p95: 0.03 ms
- p99: 0.05 ms
- mean: 0.01 ms

## Cost

- total: $0.0000
- per 1k decisions: $0.0000

## Verification-tier usage (which check made the call?)

| Tier | Count | % |
|---|---|---|
| embedding | 1196 | 59.8% |
| entity_guard | 304 | 15.2% |
| exact | 242 | 12.1% |
| policy | 214 | 10.7% |
| verify | 44 | 2.2% |
