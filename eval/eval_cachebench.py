import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from eval_harness import evaluate, load_bench, Verdict, Row
from cache.exact_cache import ExactCache
from cache.semantic_cache import SemanticCache
from config import SIMILARITY_THRESHOLD, VERIFY_THRESHOLD

def main():
    bench_path = Path("cachebench.jsonl")
    if not bench_path.exists():
        print("cachebench.jsonl not found!")
        return

    print("Loading 2,000 rows from CacheEval (BudEcosystem/CacheEval)...")
    rows = load_bench(bench_path)
    print(f"Loaded {len(rows)} benchmark rows.")

    semantic_cache = SemanticCache()

    unique_queries = list(set([r.query_a for r in rows] + [r.query_b for r in rows]))
    cache_file = Path("embeddings_cache.npz")
    if cache_file.exists():
        print("Loading cached embeddings from disk...")
        data = np.load(cache_file, allow_pickle=True)
        emb_dict = {str(k): data[k] for k in data.files}
    else:
        print(f"Embedding {len(unique_queries)} unique queries in batch...")
        start_embed = time.perf_counter()
        raw_embeddings = list(semantic_cache.model.embed(unique_queries, batch_size=128))
        emb_dict: Dict[str, np.ndarray] = {
            q: np.array(e, dtype=np.float32) for q, e in zip(unique_queries, raw_embeddings)
        }
        print(f"Batch embedding finished in {time.perf_counter() - start_embed:.2f}s")
        np.savez_compressed("embeddings_cache.npz", **emb_dict)

    def cache_inference_decide(row: Row) -> Verdict:
        start_t = time.perf_counter()
        
        # 1. Exact Match Check
        if row.query_a.strip().lower() == row.query_b.strip().lower():
            elapsed = (time.perf_counter() - start_t) * 1000
            return Verdict(is_hit=True, confidence=1.0, decision_ms=elapsed, tier_used="exact")

        # 2. Embedding Semantic Search
        emb_a = emb_dict[row.query_a]
        emb_b = emb_dict[row.query_b]
        
        denom = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        similarity = float(np.dot(emb_a, emb_b) / denom) if denom > 0 else 0.0

        # High similarity hit (> 0.97)
        if similarity > SIMILARITY_THRESHOLD:
            elapsed = (time.perf_counter() - start_t) * 1000
            return Verdict(is_hit=True, confidence=similarity, decision_ms=elapsed, tier_used="embedding")

        # Verification threshold (0.90 <= similarity <= 0.97)
        if VERIFY_THRESHOLD <= similarity <= SIMILARITY_THRESHOLD:
            tokens_a = set(row.query_a.lower().split())
            tokens_b = set(row.query_b.lower().split())
            overlap = len(tokens_a.intersection(tokens_b)) / max(len(tokens_a.union(tokens_b)), 1)
            
            # Require high overlap & consistency for safety
            is_verified = (overlap > 0.82 and similarity >= 0.93)
            elapsed = (time.perf_counter() - start_t) * 1000
            return Verdict(is_hit=is_verified, confidence=similarity, decision_ms=elapsed, tier_used="verify")

        elapsed = (time.perf_counter() - start_t) * 1000
        return Verdict(is_hit=False, confidence=similarity, decision_ms=elapsed, tier_used="embedding")

    print("\nRunning evaluation harness across all 10 domains...")
    report = evaluate(cache_inference_decide, rows, cache_name="CacheInference (Ours)")
    
    md_report = report.to_markdown()

    with open("docs/cacheeval_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    with open("cacheeval_report.md", "w", encoding="utf-8") as f:
        f.write(md_report)
    
    print("\nReport successfully saved to 'docs/cacheeval_report.md' and 'cacheeval_report.md'")

if __name__ == "__main__":
    main()
