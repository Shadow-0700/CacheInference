# CacheInference — Benchmark Results

## Performance Comparison Table

| System                |   Total Requests | Hit Rate (%)   | False Positive Rate (%)   |   Avg Latency (ms) | Est. Cost Saved (%)   |
|-----------------------|------------------|----------------|---------------------------|--------------------|-----------------------|
| No-Cache Baseline     |               13 | 0.0%           | 0.0%                      |             350.42 | 0.0%                  |
| CacheInference (Ours) |               13 | 46.2%          | 0.0%                      |             241.28 | 46.2%                 |
| GPTCache (Standard)   |               13 | 38.5%          | 7.7%                      |             228.4  | 38.5%                 |
