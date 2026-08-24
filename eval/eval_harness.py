#!/usr/bin/env python3
"""
CacheBench v1 — evaluation harness.

Given a cache implementation (any class exposing `decide(row) -> Verdict`), this
runs every row through the cache, compares against the ground-truth label, and
emits per-domain and overall metrics.

Usage:
    python eval_harness.py --cache-module my_cache --cache-class MyCache \
        --bench /path/to/cachebench.jsonl --out report.md

Or import as a library:
    from eval_harness import evaluate
    report = evaluate(my_cache, bench_path)
    print(report.to_markdown())
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Verdict & ground truth
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """A cache's decision for a single pair."""
    is_hit: bool                          # True = cache returns query_a's answer for query_b
    confidence: float = 0.0               # optional, for PR/ROC analysis
    decision_ms: float = 0.0              # wall-clock latency of the decision
    tier_used: str = "unknown"            # "exact" | "intent" | "embedding" | "judge" | etc.
    cost_usd: float = 0.0                 # incremental cost of the decision (e.g. judge call)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Row:
    """One CacheBench row."""
    id: str
    domain: str
    subcategory: str
    label: str                            # EQUIV | PARA_SAFE | RELATED_UNSAFE | ADVERSARIAL | UNRELATED
    binary_label: str                     # HIT | MISS
    difficulty: str                       # easy | medium | hard
    query_a: str
    query_b: str
    context_a: Optional[str] = None
    context_b: Optional[str] = None
    system_prompt_a: Optional[str] = None
    system_prompt_b: Optional[str] = None
    tools_a: Optional[Any] = None
    tools_b: Optional[Any] = None
    expected_answer_a: Optional[str] = None
    expected_answer_b: Optional[str] = None
    rationale: str = ""
    verification_method: str = "exact_match"
    verification_payload: Any = None
    construction_method: str = ""
    source: str = ""
    real_world_grounding: str = ""

    @property
    def truth_is_hit(self) -> bool:
        return self.binary_label.upper() == "HIT"


def load_bench(path: Path) -> List[Row]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} JSON parse error: {e}") from e
            # Defensive checks on the load path — these prevent silent miscount.
            for required in ("id", "domain", "label", "binary_label", "difficulty",
                             "query_a", "query_b"):
                if d.get(required) is None:
                    raise ValueError(f"{path}:{ln} row missing required field {required!r}")
            if d["binary_label"] not in {"HIT", "MISS"}:
                raise ValueError(
                    f"{path}:{ln} invalid binary_label {d['binary_label']!r} "
                    f"(must be HIT or MISS)"
                )
            rows.append(Row(**{k: d.get(k) for k in Row.__dataclass_fields__}))
    return rows


# ---------------------------------------------------------------------------
# Confusion matrix and metrics
# ---------------------------------------------------------------------------

@dataclass
class ConfusionMatrix:
    tp: int = 0   # truth=HIT, cache=HIT
    fp: int = 0   # truth=MISS, cache=HIT (the safety-critical failure mode)
    fn: int = 0   # truth=HIT, cache=MISS
    tn: int = 0   # truth=MISS, cache=MISS

    def add(self, truth_is_hit: bool, predicted_hit: bool) -> None:
        if truth_is_hit and predicted_hit:
            self.tp += 1
        elif truth_is_hit and not predicted_hit:
            self.fn += 1
        elif (not truth_is_hit) and predicted_hit:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_hit_rate(self) -> float:
        """FP / (FP + TN). The safety-critical number for a cache."""
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": self.total,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_hit_rate": round(self.false_hit_rate, 4),
            "accuracy": round(self.accuracy, 4),
        }


# ---------------------------------------------------------------------------
# Latency / cost stats
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Wilson CI (95%) for binomial proportion — for confidence intervals
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class Report:
    cache_name: str
    overall: ConfusionMatrix
    by_domain: Dict[str, ConfusionMatrix]
    by_label: Dict[str, ConfusionMatrix]   # restricted to MISS labels for FHR breakdown
    by_difficulty: Dict[str, ConfusionMatrix]
    by_tier: Dict[str, int]                # how many decisions used each tier
    latency_ms: List[float]
    cost_usd_per_decision: List[float]
    total_cost_usd: float

    def to_markdown(self) -> str:
        lo, hi = wilson_ci(self.overall.tp + self.overall.tn, self.overall.total)
        lines = [
            f"# CacheBench v1 — Results for `{self.cache_name}`",
            "",
            "## Headline (binary HIT/MISS classification)",
            "",
            f"- **Precision:** {self.overall.precision:.4f}",
            f"- **Recall:** {self.overall.recall:.4f}",
            f"- **F1:** {self.overall.f1:.4f}",
            f"- **False-Hit Rate (FHR):** {self.overall.false_hit_rate:.4f}  ← safety-critical",
            f"- **Accuracy:** {self.overall.accuracy:.4f}  (95% Wilson CI: {lo:.3f}, {hi:.3f})",
            f"- **N:** {self.overall.total}",
            "",
            "## Confusion matrix (overall)",
            "",
            "| | predicted HIT | predicted MISS |",
            "|---|---|---|",
            f"| truth HIT  | TP={self.overall.tp} | FN={self.overall.fn} |",
            f"| truth MISS | FP={self.overall.fp} | TN={self.overall.tn} |",
            "",
            "## Per-domain breakdown",
            "",
            "| Domain | N | F1 | FHR | Precision | Recall |",
            "|---|---|---|---|---|---|",
        ]
        for dom in sorted(self.by_domain):
            cm = self.by_domain[dom]
            lines.append(
                f"| {dom} | {cm.total} | {cm.f1:.3f} | **{cm.false_hit_rate:.3f}** | "
                f"{cm.precision:.3f} | {cm.recall:.3f} |"
            )
        lines += [
            "",
            "## Per-label FHR (false-hit rate by label class — the safety dimensions)",
            "",
            "| Label | N | FP | FHR |",
            "|---|---|---|---|",
        ]
        for lab in ["RELATED_UNSAFE", "ADVERSARIAL", "UNRELATED"]:
            cm = self.by_label.get(lab, ConfusionMatrix())
            lines.append(f"| {lab} | {cm.total} | {cm.fp} | {cm.false_hit_rate:.3f} |")
        lines += [
            "",
            "## Per-difficulty breakdown",
            "",
            "| Difficulty | N | F1 | FHR |",
            "|---|---|---|---|",
        ]
        for d in ["easy", "medium", "hard"]:
            cm = self.by_difficulty.get(d, ConfusionMatrix())
            lines.append(f"| {d} | {cm.total} | {cm.f1:.3f} | {cm.false_hit_rate:.3f} |")
        lines += [
            "",
            "## Latency",
            "",
            f"- p50: {percentile(self.latency_ms, 0.5):.2f} ms",
            f"- p95: {percentile(self.latency_ms, 0.95):.2f} ms",
            f"- p99: {percentile(self.latency_ms, 0.99):.2f} ms",
            f"- mean: {mean(self.latency_ms) if self.latency_ms else 0:.2f} ms",
            "",
            "## Cost",
            "",
            f"- total: ${self.total_cost_usd:.4f}",
            f"- per 1k decisions: ${self.total_cost_usd / max(1, self.overall.total) * 1000:.4f}",
            "",
            "## Verification-tier usage (which check made the call?)",
            "",
            "| Tier | Count | % |",
            "|---|---|---|",
        ]
        total = sum(self.by_tier.values()) or 1
        for tier, count in sorted(self.by_tier.items(), key=lambda x: -x[1]):
            lines.append(f"| {tier} | {count} | {100 * count / total:.1f}% |")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(cache_decide: Callable[[Row], Verdict], rows: Iterable[Row],
             cache_name: str = "unknown", on_error: str = "miss") -> Report:
    """Evaluate ``cache_decide`` against ``rows``.

    ``on_error``: how to handle a cache that raises or returns garbage.
      * ``"miss"`` (default): record a Verdict(is_hit=False, tier_used="error")
        and continue. Errors are visible in the tier report; the run completes.
      * ``"raise"``: re-raise — useful in CI / debugging.
    """
    rows = list(rows)
    overall = ConfusionMatrix()
    by_domain: Dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)
    by_label: Dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)
    by_diff: Dict[str, ConfusionMatrix] = defaultdict(ConfusionMatrix)
    by_tier: Dict[str, int] = defaultdict(int)
    latencies: List[float] = []
    costs: List[float] = []
    total_cost = 0.0
    error_count = 0

    for row in rows:
        t0 = time.perf_counter()
        try:
            verdict = cache_decide(row)
            if not isinstance(verdict, Verdict):
                raise TypeError(f"cache must return Verdict, got {type(verdict).__name__}")
        except Exception as exc:
            if on_error == "raise":
                raise
            error_count += 1
            verdict = Verdict(is_hit=False, tier_used="error",
                              extra={"error": f"{type(exc).__name__}: {exc}"})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # use the cache's reported latency if positive, else wall-clock
        reported = verdict.decision_ms if verdict.decision_ms is not None else 0.0
        lat = reported if reported > 0 else elapsed_ms
        latencies.append(max(0.0, lat))
        cost = verdict.cost_usd if verdict.cost_usd is not None else 0.0
        costs.append(max(0.0, cost))
        total_cost += max(0.0, cost)
        by_tier[verdict.tier_used or "unknown"] += 1

        overall.add(row.truth_is_hit, bool(verdict.is_hit))
        by_domain[row.domain].add(row.truth_is_hit, bool(verdict.is_hit))
        by_label[row.label].add(row.truth_is_hit, bool(verdict.is_hit))
        by_diff[row.difficulty].add(row.truth_is_hit, bool(verdict.is_hit))

    if error_count:
        print(f"WARNING: {error_count} cache errors encountered (counted as MISS).")

    return Report(
        cache_name=cache_name,
        overall=overall,
        by_domain=dict(by_domain),
        by_label=dict(by_label),
        by_difficulty=dict(by_diff),
        by_tier=dict(by_tier),
        latency_ms=latencies,
        cost_usd_per_decision=costs,
        total_cost_usd=total_cost,
    )


# ---------------------------------------------------------------------------
# Baseline caches (built-in references)
# ---------------------------------------------------------------------------

class AlwaysMissCache:
    """No-op cache — always returns MISS. Establishes the floor."""
    def __call__(self, row: Row) -> Verdict:
        return Verdict(is_hit=False, tier_used="always_miss")


class AlwaysHitCache:
    """Always returns HIT. Establishes the maximum-recall, worst-FHR ceiling."""
    def __call__(self, row: Row) -> Verdict:
        return Verdict(is_hit=True, tier_used="always_hit")


class ExactMatchCache:
    """Hits only if query_a == query_b verbatim. Useful sanity check."""
    def __call__(self, row: Row) -> Verdict:
        return Verdict(is_hit=row.query_a == row.query_b, tier_used="exact_match")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="CacheBench v1 evaluation harness")
    ap.add_argument("--bench", type=Path, required=True,
                    help="Path to cachebench.jsonl (the merged 2000-row file)")
    ap.add_argument("--cache-module", type=str, default=None,
                    help="Python module exposing `cache` callable (Row -> Verdict)")
    ap.add_argument("--cache-class", type=str, default=None,
                    help="Class name within the module to instantiate")
    ap.add_argument("--cache-name", type=str, default=None,
                    help="Friendly name for the report header")
    ap.add_argument("--baseline", choices=["always_miss", "always_hit", "exact_match"],
                    help="Use a built-in baseline instead of loading a cache")
    ap.add_argument("--out", type=Path, default=Path("report.md"),
                    help="Where to write the markdown report")
    args = ap.parse_args()

    rows = load_bench(args.bench)

    if args.baseline:
        name = args.cache_name or args.baseline
        cache = {
            "always_miss": AlwaysMissCache(),
            "always_hit": AlwaysHitCache(),
            "exact_match": ExactMatchCache(),
        }[args.baseline]
    else:
        if not args.cache_module:
            ap.error("Need --baseline or --cache-module")
        mod = importlib.import_module(args.cache_module)
        if args.cache_class:
            cache = getattr(mod, args.cache_class)()
        else:
            cache = getattr(mod, "cache")
        name = args.cache_name or args.cache_module

    report = evaluate(cache, rows, cache_name=name)
    args.out.write_text(report.to_markdown())
    print(report.to_markdown())


if __name__ == "__main__":
    main()
