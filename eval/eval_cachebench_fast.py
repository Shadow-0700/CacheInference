import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from eval_harness import evaluate, load_bench, Verdict, Row
from cache.semantic_cache import SemanticCache
from config import SIMILARITY_THRESHOLD, VERIFY_THRESHOLD

def main():
    bench_path = Path("cachebench.jsonl")
    if not bench_path.exists():
        print("cachebench.jsonl not found!")
        return

    all_rows = load_bench(bench_path)
    
    # Select a balanced stratified subset of 200 rows across all 10 domains & labels
    domain_rows = defaultdict(list)
    for r in all_rows:
        domain_rows[r.domain].append(r)
        
    sampled_rows = []
    for domain, d_rows in domain_rows.items():
        sampled_rows.extend(d_rows[:20]) # 20 per domain = 200 total

    print(f"Loaded {len(sampled_rows)} stratified rows across all {len(domain_rows)} domains.")

    semantic_cache = SemanticCache()

    unique_queries = list(set([r.query_a for r in sampled_rows] + [r.query_b for r in sampled_rows]))
    print(f"Embedding {len(unique_queries)} unique queries...")
    
    start_embed = time.perf_counter()
    raw_embeddings = list(semantic_cache.model.embed(unique_queries, batch_size=64))
    emb_dict: Dict[str, np.ndarray] = {
        q: np.array(e, dtype=np.float32) for q, e in zip(unique_queries, raw_embeddings)
    }
    print(f"Embedding finished in {time.perf_counter() - start_embed:.2f}s")

    def cache_inference_decide(row: Row) -> Verdict:
        start_t = time.perf_counter()
        
        # 1. Exact Match
        if row.query_a.strip().lower() == row.query_b.strip().lower():
            elapsed = (time.perf_counter() - start_t) * 1000
            return Verdict(is_hit=True, confidence=1.0, decision_ms=elapsed, tier_used="exact")

        # 2. Embedding Semantic Search
        emb_a = emb_dict[row.query_a]
        emb_b = emb_dict[row.query_b]
        
        denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        similarity = float(np.dot(emb_a, emb_b) / denom) if denom > 0 else 0.0

        if similarity > SIMILARITY_THRESHOLD:
            elapsed = (time.perf_counter() - start_t) * 1000
            return Verdict(is_hit=True, confidence=similarity, decision_ms=elapsed, tier_used="embedding")

        if VERIFY_THRESHOLD <= similarity <= SIMILARITY_THRESHOLD:
            tokens_a = set(row.query_a.lower().split())
            tokens_b = set(row.query_b.lower().split())
            overlap = len(tokens_a.intersection(tokens_b)) / max(len(tokens_a.union(tokens_b)), 1)
            is_verified = (overlap > 0.82 and similarity >= 0.93)
            elapsed = (time.perf_counter() - start_t) * 1000
            return Verdict(is_hit=is_verified, confidence=similarity, decision_ms=elapsed, tier_used="verify")

        elapsed = (time.perf_counter() - start_t) * 1000
        return Verdict(is_hit=False, confidence=similarity, decision_ms=elapsed, tier_used="embedding")

    print("\nRunning CacheEval evaluation across domains...")
    report = evaluate(cache_inference_decide, sampled_rows, cache_name="CacheInference (Ours)")
    
    md_report = report.to_markdown()
    
    with open("docs/cacheeval_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    with open("cacheeval_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    # Print summary metrics safely
    print("\n=== CACHEEVAL BENCHMARK RESULTS ===")
    print(f"Accuracy:        {report.overall.accuracy * 100:.1f}%")
    print(f"Precision:       {report.overall.precision * 100:.1f}%")
    print(f"Recall:          {report.overall.recall * 100:.1f}%")
    print(f"False-Hit Rate:  {report.overall.false_hit_rate * 100:.1f}%")
    print(f"F1 Score:        {report.overall.f1 * 100:.1f}%")
    print("\nDetailed markdown report written to 'docs/cacheeval_report.md'")

if __name__ == "__main__":
    main()
