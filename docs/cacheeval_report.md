# CacheBench v1 — Results for `CacheInference (LLM-Verified)`

## Headline (binary HIT/MISS classification)

- **Precision:** 0.6645
- **Recall:** 0.9007
- **F1:** 0.7648
- **False-Hit Rate (FHR):** 0.2317  ← safety-critical
- **Accuracy:** 0.8130  (95% Wilson CI: 0.795, 0.829)
- **N:** 2000

## Confusion matrix (overall)

| | predicted HIT | predicted MISS |
|---|---|---|
| truth HIT  | TP=608 | FN=67 |
| truth MISS | FP=307 | TN=1018 |

## Per-domain breakdown

| Domain | N | F1 | FHR | Precision | Recall |
|---|---|---|---|---|---|
| code | 160 | 0.929 | **0.080** | 0.881 | 0.983 |
| conversational | 260 | 0.811 | **0.357** | 0.699 | 0.967 |
| creative | 180 | 0.000 | **0.244** | 0.000 | 0.000 |
| math | 160 | 0.975 | **0.020** | 0.967 | 0.983 |
| multi_turn | 180 | 0.655 | **0.425** | 0.514 | 0.900 |
| multilingual | 150 | 0.453 | **0.080** | 0.680 | 0.340 |
| personalized | 140 | 0.000 | **0.579** | 0.000 | 0.000 |
| qa_factual | 230 | 0.906 | **0.038** | 0.946 | 0.870 |
| qa_open | 260 | 0.954 | **0.036** | 0.958 | 0.950 |
| tool | 280 | 0.785 | **0.303** | 0.658 | 0.971 |

## Per-label FHR (false-hit rate by label class — the safety dimensions)

| Label | N | FP | FHR |
|---|---|---|---|
| RELATED_UNSAFE | 420 | 152 | 0.362 |
| ADVERSARIAL | 410 | 66 | 0.161 |
| UNRELATED | 340 | 1 | 0.003 |

## Per-difficulty breakdown

| Difficulty | N | F1 | FHR |
|---|---|---|---|
| easy | 1184 | 0.861 | 0.113 |
| medium | 238 | 0.851 | 0.146 |
| hard | 578 | 0.614 | 0.554 |

## Latency

- p50: 24.30 ms
- p95: 36.30 ms
- p99: 37.53 ms
- mean: 17.52 ms

## Cost

- total: $0.0411
- per 1k decisions: $0.0205

## Verification-tier usage (which check made the call?)

| Tier | Count | % |
|---|---|---|
| verify | 1173 | 58.6% |
| embedding | 585 | 29.2% |
| exact | 242 | 12.1% |
