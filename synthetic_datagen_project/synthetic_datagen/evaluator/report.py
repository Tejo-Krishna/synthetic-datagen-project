"""
evaluator/report.py
-------------------
Aggregation and console reporting for the evaluate pipeline.

EvaluationReport holds aggregate metrics across all scored conversations.
generate_report() builds it from a list of evaluated records.
print_report() formats it for console output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationReport:
    """
    Aggregated evaluation metrics across all scored conversations.

    Attributes:
        total:               Total number of records evaluated.
        passed:              Records that passed all quality gates.
        failed:              Records that failed (including after repair).
        judge_errors:        Records where judge scoring itself failed.
        repaired:            Records that were repaired and now pass.
        repair_attempted:    Records where repair was attempted.
        mean_tool_correctness: Mean score across all scoreable records.
        mean_task_completion:  Mean score across all scoreable records.
        mean_naturalness:      Mean score across all scoreable records.
        mean_overall:          Mean of means across all scoreable records.
        pass_rate:             Fraction of total that passed.
        by_domain:             Per-domain pass rate and mean scores.
        by_pattern:            Per-pattern pass rate and mean scores.
        threshold_mean:        The mean threshold used (for CI assertion).
        threshold_tool_correctness: Hard floor used.
        threshold_task_completion:  Hard floor used.
        threshold_naturalness:      Hard floor used.
    """
    total: int = 0
    passed: int = 0
    failed: int = 0
    judge_errors: int = 0
    repaired: int = 0
    repair_attempted: int = 0

    mean_tool_correctness: float | None = None
    mean_task_completion: float | None = None
    mean_naturalness: float | None = None
    mean_overall: float | None = None
    pass_rate: float = 0.0

    by_domain: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_pattern: dict[str, dict[str, Any]] = field(default_factory=dict)

    threshold_mean: float = 3.5
    threshold_tool_correctness: float = 3.5
    threshold_task_completion: float = 3.5
    threshold_naturalness: float = 3.0


def generate_report(
    records: list[dict],
    threshold_mean: float = 3.5,
    threshold_tool_correctness: float = 3.5,
    threshold_task_completion: float = 3.5,
    threshold_naturalness: float = 3.0,
) -> EvaluationReport:
    """
    Build an EvaluationReport from a list of evaluated records.

    Each record is expected to have a 'judge_scores' dict and a 'passed' bool,
    as written by scorer.attach_scores().
    """
    report = EvaluationReport(
        total=len(records),
        threshold_mean=threshold_mean,
        threshold_tool_correctness=threshold_tool_correctness,
        threshold_task_completion=threshold_task_completion,
        threshold_naturalness=threshold_naturalness,
    )

    tc_scores: list[float] = []
    comp_scores: list[float] = []
    nat_scores: list[float] = []

    domain_buckets: dict[str, list[dict]] = {}
    pattern_buckets: dict[str, list[dict]] = {}

    for rec in records:
        scores = rec.get("judge_scores", {})
        passed = rec.get("passed", False)
        meta = rec.get("metadata", {})

        domain = meta.get("domain", "unknown")
        pattern = meta.get("pattern_type", "unknown")

        # Judge error
        if scores.get("error") or scores.get("tool_correctness") is None:
            report.judge_errors += 1
            report.failed += 1
        elif passed:
            report.passed += 1
        else:
            report.failed += 1

        # Repair tracking
        if meta.get("repair_attempts", 0) > 0:
            report.repair_attempted += 1
            if passed and meta.get("repair_attempts", 0) > 0:
                report.repaired += 1

        # Accumulate scores
        tc = scores.get("tool_correctness")
        comp = scores.get("task_completion")
        nat = scores.get("naturalness")
        if tc is not None:
            tc_scores.append(tc)
        if comp is not None:
            comp_scores.append(comp)
        if nat is not None:
            nat_scores.append(nat)

        # Domain bucket
        domain_buckets.setdefault(domain, []).append(rec)
        pattern_buckets.setdefault(pattern, []).append(rec)

    # Compute means
    if tc_scores:
        report.mean_tool_correctness = round(sum(tc_scores) / len(tc_scores), 4)
    if comp_scores:
        report.mean_task_completion = round(sum(comp_scores) / len(comp_scores), 4)
    if nat_scores:
        report.mean_naturalness = round(sum(nat_scores) / len(nat_scores), 4)

    scoreable = [s for s in [report.mean_tool_correctness,
                              report.mean_task_completion,
                              report.mean_naturalness] if s is not None]
    if scoreable:
        report.mean_overall = round(sum(scoreable) / len(scoreable), 4)

    if report.total > 0:
        report.pass_rate = round(report.passed / report.total, 4)

    # Per-domain breakdown
    for domain, recs in domain_buckets.items():
        report.by_domain[domain] = _bucket_stats(recs)

    # Per-pattern breakdown
    for pattern, recs in pattern_buckets.items():
        report.by_pattern[pattern] = _bucket_stats(recs)

    return report


def _bucket_stats(recs: list[dict]) -> dict[str, Any]:
    """Compute pass rate and mean scores for a bucket of records."""
    total = len(recs)
    passed = sum(1 for r in recs if r.get("passed", False))
    tc_vals = [r["judge_scores"]["tool_correctness"]
               for r in recs
               if r.get("judge_scores", {}).get("tool_correctness") is not None]
    comp_vals = [r["judge_scores"]["task_completion"]
                 for r in recs
                 if r.get("judge_scores", {}).get("task_completion") is not None]
    nat_vals = [r["judge_scores"]["naturalness"]
                for r in recs
                if r.get("judge_scores", {}).get("naturalness") is not None]

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
        "mean_tool_correctness": round(sum(tc_vals) / len(tc_vals), 4) if tc_vals else None,
        "mean_task_completion": round(sum(comp_vals) / len(comp_vals), 4) if comp_vals else None,
        "mean_naturalness": round(sum(nat_vals) / len(nat_vals), 4) if nat_vals else None,
    }


def print_report(report: EvaluationReport) -> None:
    """Print a formatted evaluation report to stdout."""
    w = 60
    print("\n" + "=" * w)
    print("  EVALUATION REPORT")
    print("=" * w)

    print(f"\n  Records evaluated:     {report.total}")
    print(f"  Passed:                {report.passed}  ({100 * report.pass_rate:.1f}%)")
    print(f"  Failed:                {report.failed}")
    print(f"  Judge errors:          {report.judge_errors}")
    if report.repair_attempted > 0:
        print(f"  Repair attempted:      {report.repair_attempted}")
        print(f"  Repaired (now pass):   {report.repaired}")

    print(f"\n  --- Mean Scores (1.0–5.0) ---")
    print(f"  tool_correctness:  {_fmt(report.mean_tool_correctness)}  "
          f"(threshold >= {report.threshold_tool_correctness})")
    print(f"  task_completion:   {_fmt(report.mean_task_completion)}  "
          f"(threshold >= {report.threshold_task_completion})")
    print(f"  naturalness:       {_fmt(report.mean_naturalness)}  "
          f"(threshold >= {report.threshold_naturalness})")
    print(f"  overall mean:      {_fmt(report.mean_overall)}  "
          f"(threshold >= {report.threshold_mean})")

    # CI assertion check
    passed_assertion = (
        report.mean_overall is not None
        and report.mean_overall >= report.threshold_mean
        and (report.mean_tool_correctness or 0) >= report.threshold_tool_correctness
        and (report.mean_task_completion or 0) >= report.threshold_task_completion
        and (report.mean_naturalness or 0) >= report.threshold_naturalness
    )
    assertion_label = "PASS" if passed_assertion else "FAIL"
    print(f"\n  Quality assertion:     {assertion_label}")

    if report.by_domain:
        print(f"\n  --- By Domain ---")
        for domain, stats in sorted(report.by_domain.items()):
            print(
                f"  {domain:<20} "
                f"pass={stats['passed']}/{stats['total']}  "
                f"mean_tc={_fmt(stats['mean_tool_correctness'])}  "
                f"mean_comp={_fmt(stats['mean_task_completion'])}  "
                f"mean_nat={_fmt(stats['mean_naturalness'])}"
            )

    if report.by_pattern:
        print(f"\n  --- By Pattern ---")
        for pattern, stats in sorted(report.by_pattern.items()):
            print(
                f"  {pattern:<28} "
                f"pass={stats['passed']}/{stats['total']}  "
                f"pass_rate={100 * stats['pass_rate']:.1f}%"
            )

    print("\n" + "=" * w + "\n")


def _fmt(val: float | None) -> str:
    """Format a score for display."""
    return f"{val:.4f}" if val is not None else "  N/A "
