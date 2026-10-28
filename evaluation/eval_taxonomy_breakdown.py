"""
Verification & Acceptance Test Suite for Taxonomy Breakdown Comparison:
Vector Chunking (Full) vs. Page Index (Full).

Focuses strictly on the direct comparison between:
- Vector Chunking (Full 100 queries)
- Page Index (Full 100 queries)

Displays clean side-by-side comparison tables across all 12 Taxonomy Groups
without displaying baseline ablation columns (Nomic, Qwen Dense, Hybrid BM25).

Usage:
    pytest evaluation/eval_taxonomy_breakdown.py -v -s
    python -m evaluation.eval_taxonomy_breakdown
    make eval-taxonomy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.eval_common import WORKSPACE_ROOT, ensure_runtime

ensure_runtime()

RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "results"
VECTOR_RESULTS_PATH = RESULTS_DIR / "vector_chunking" / "benchmark_results.json"
PAGE_RESULTS_PATH = RESULTS_DIR / "page_index" / "benchmark_results.json"
TABLE_WIDTH = 118


def load_backend_benchmark(path: Path, name: str) -> dict[str, Any]:
    """Load benchmark results JSON file for a given backend."""
    if not path.is_file():
        pytest.skip(f"Benchmark results for '{name}' not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_comparison_table(
    vector_data: dict[str, Any],
    page_data: dict[str, Any],
) -> str:
    """Format clean side-by-side Table 2 comparing Vector Chunking Full vs Page Index Full."""
    v_hr = vector_data["ablation"]["hybrid_rerank"]
    p_hr = page_data["ablation"]["hybrid_rerank"]

    v_groups = v_hr.get("group_breakdown", {})
    p_groups = p_hr.get("group_breakdown", {})

    lines: list[str] = [
        "\n" + "=" * TABLE_WIDTH,
        "  TABLE 2 — TAXONOMY COMPARISON: VECTOR CHUNKING (FULL) VS PAGE INDEX (FULL) [12 GROUPS / 100 QUERIES]",
        "=" * TABLE_WIDTH,
        f"{'Evaluation Taxonomy Group':<42} | {'Queries':<7} | {'Vector Chunking':<17} | {'Page Index':<17} | {'VC MRR@3':<9} | {'PI MRR@3':<8}",
        "-" * TABLE_WIDTH,
    ]

    total_q_sum = 0
    v_pass_sum = 0
    p_pass_sum = 0

    for g_name, v_stats in v_groups.items():
        p_stats = p_groups.get(g_name, {})
        q_count = v_stats.get("total_queries", 0)
        v_passed = v_stats.get("passed_queries", 0)
        p_passed = p_stats.get("passed_queries", 0)

        total_q_sum += q_count
        v_pass_sum += v_passed
        p_pass_sum += p_passed

        v_rate = f"{v_stats.get('assertion_pass_rate', 0.0):>5.1f}% ({v_passed}/{q_count})"
        p_rate = f"{p_stats.get('assertion_pass_rate', 0.0):>5.1f}% ({p_passed}/{q_count})"
        v_mrr = f"{v_stats.get('mrr_at_3', 0.0):>7.4f}"
        p_mrr = f"{p_stats.get('mrr_at_3', 0.0):>7.4f}"

        lines.append(
            f"{g_name[:42]:<42} | {q_count:>7} | {v_rate:<17} | {p_rate:<17} | {v_mrr:<9} | {p_mrr:<8}"
        )

    lines.append("-" * TABLE_WIDTH)
    v_overall_rate = f"{(v_pass_sum / total_q_sum * 100.0) if total_q_sum else 0.0:>5.1f}% ({v_pass_sum}/{total_q_sum})"
    p_overall_rate = f"{(p_pass_sum / total_q_sum * 100.0) if total_q_sum else 0.0:>5.1f}% ({p_pass_sum}/{total_q_sum})"
    v_total_mrr = f"{v_hr.get('mrr_at_3', 0.0):>7.4f}"
    p_total_mrr = f"{p_hr.get('mrr_at_3', 0.0):>7.4f}"

    lines.append(
        f"{'OVERALL STRICT ACCURACY':<42} | {total_q_sum:>7} | {v_overall_rate:<17} | {p_overall_rate:<17} | {v_total_mrr:<9} | {p_total_mrr:<8}"
    )
    lines.append("=" * TABLE_WIDTH)

    # Summary metrics block
    lines.append("\nSummary Key Takeaways:")
    lines.append(f"  • Vector Chunking (Full): {v_pass_sum}/{total_q_sum} passed ({v_overall_rate.strip()}), MRR@3 = {v_total_mrr.strip()}, Latency = {v_hr.get('avg_latency_ms', 0.0):.1f} ms")
    lines.append(f"  • Page Index (Full):      {p_pass_sum}/{total_q_sum} passed ({p_overall_rate.strip()}), MRR@3 = {p_total_mrr.strip()}, Latency = {p_hr.get('avg_latency_ms', 0.0):.1f} ms")

    return "\n".join(lines)


def test_taxonomy_comparison_vector_chunking_vs_page_index() -> None:
    """Automated test validating that both backends cover all 12 groups with >=90% pass rate."""
    v_data = load_backend_benchmark(VECTOR_RESULTS_PATH, "vector_chunking")
    p_data = load_backend_benchmark(PAGE_RESULTS_PATH, "page_index")

    v_hr = v_data["ablation"]["hybrid_rerank"]
    p_hr = p_data["ablation"]["hybrid_rerank"]

    v_groups = v_hr["group_breakdown"]
    p_groups = p_hr["group_breakdown"]

    # Both must have 12 groups
    assert len(v_groups) == 12, f"Vector chunking expected 12 groups, found {len(v_groups)}"
    assert len(p_groups) == 12, f"Page index expected 12 groups, found {len(p_groups)}"

    # Both must cover 100 queries
    assert sum(g["total_queries"] for g in v_groups.values()) == 100
    assert sum(g["total_queries"] for g in p_groups.values()) == 100

    # Both must achieve high strict pass rate >= 90%
    assert v_hr["assertion_pass_rate"] >= 90.0, f"Vector chunking pass rate {v_hr['assertion_pass_rate']}% < 90%"
    assert p_hr["assertion_pass_rate"] >= 90.0, f"Page index pass rate {p_hr['assertion_pass_rate']}% < 90%"

    # Output comparison table
    print(format_comparison_table(v_data, p_data))


def main() -> None:
    """CLI entrypoint to render the comparison table directly."""
    v_data = load_backend_benchmark(VECTOR_RESULTS_PATH, "vector_chunking")
    p_data = load_backend_benchmark(PAGE_RESULTS_PATH, "page_index")
    print(format_comparison_table(v_data, p_data))


if __name__ == "__main__":
    main()
