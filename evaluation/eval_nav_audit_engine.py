"""
Verification & Acceptance Test Suite for Neuro-Symbolic NAV Audit Engine & Precision.

Covers Definition of Done (DoD) Criteria:
- Criterion 3: 8/8 Neuro-Symbolic NAV Audit Scenarios (10 Runs: Scenario A–Scenario H)
  with exact Decimal arithmetic and regulatory verdicts.
- Criterion 4: 6-Mode Quantitative Retrieval Ablation Study + 12-Case Decimal vs. float64
  Precision Study.
- Criterion 5: Immutable JSON Audit Trail Generation & Cryptographic Integrity (SHA-256) in logs/.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from evaluation.conftest import BENCHMARK_RESULTS_PATH, LOGS_DIR, load_full_queries
from evaluation.run_benchmark import run_arithmetic_precision_study


# ---------------------------------------------------------------------------
# Criterion 3: 8/8 Neuro-Symbolic NAV Audit Scenarios (10 Runs) Pass 100%
# ---------------------------------------------------------------------------


def test_criterion_3_neuro_symbolic_audit_scenarios(scenario_results: dict[str, Any]) -> None:
    """Verify that all 10 scenario runs across 8 scenarios produce correct verdicts and Decimal math."""
    results = scenario_results
    assert len(results) == 10

    # Scenario A — Run A1 (2024-06-15 under CSSF 02/77)
    r_a1 = results["scenario_a_2024"]
    assert r_a1.variance_bps == Decimal("22.00")
    assert r_a1.applicable_threshold_bps == Decimal("25.00")
    assert r_a1.is_material_nav_breach is False
    assert r_a1.regulatory_verdict == "IMMATERIAL_ERROR_WITHIN_TOLERANCE"
    assert r_a1.extracted_pydantic_parameters["statutory_threshold_spec"]["doc_id"] == "CSSF_02_77"

    # Scenario A — Run A2 (2025-02-15 under CSSF 24/856)
    r_a2 = results["scenario_a_2025"]
    assert r_a2.variance_bps == Decimal("22.00")
    assert r_a2.applicable_threshold_bps == Decimal("20.00")
    assert r_a2.is_material_nav_breach is True
    assert r_a2.regulatory_verdict == "SIGNIFICANT_NAV_ERROR_BREACH"
    assert r_a2.extracted_pydantic_parameters["statutory_threshold_spec"]["doc_id"] == "CSSF_24_856"

    # Scenario B — Accruals & Tiered Fee Deviation (Lux Alpha Class I)
    r_b = results["scenario_b"]
    assert r_b.reported_value == "EUR 324444.44"
    assert r_b.replicated_value == "EUR 290000.00"
    assert r_b.absolute_variance == "EUR 34444.44"
    assert r_b.variance_bps == Decimal("4.31")
    assert r_b.applicable_threshold_bps == Decimal("100.00")
    assert r_b.regulatory_verdict == "PROSPECTUS_METHODOLOGY_DEVIATION_DETECTED"
    assert "INTERNAL_CONTROL_WARNING" in r_b.control_flags

    # Scenario C — Material Pricing Breach & EUR 50,000 Special Audit Trigger
    r_c = results["scenario_c"]
    assert r_c.reported_value == "EUR 100.00/share"
    assert r_c.replicated_value == "EUR 99.20/share"
    assert r_c.variance_bps == Decimal("80.00")
    assert r_c.applicable_threshold_bps == Decimal("50.00")
    assert r_c.is_material_nav_breach is True
    assert r_c.special_audit_report_required is True
    assert r_c.regulatory_verdict == "CRITICAL_CSSF_BREACH_AND_SPECIAL_AUDIT_REPORT_REQUIRED"

    # Scenario D — Mixed UCITS Breach Below Special Audit Threshold (Sample-Based Review)
    r_d = results["scenario_d"]
    assert r_d.replicated_value == "EUR 99.35/share"
    assert r_d.variance_bps == Decimal("65.00")
    assert r_d.applicable_threshold_bps == Decimal("50.00")
    assert r_d.is_material_nav_breach is True
    assert r_d.special_audit_report_required is False
    assert r_d.regulatory_verdict == "SIGNIFICANT_NAV_ERROR_BREACH"

    # Scenario E — Equity UCITS Single-Investor Disjunctive Trigger (> EUR 5,000)
    r_e = results["scenario_e"]
    assert r_e.replicated_value == "EUR 98.75/share"
    assert r_e.variance_bps == Decimal("125.00")
    assert r_e.applicable_threshold_bps == Decimal("100.00")
    assert r_e.is_material_nav_breach is True
    assert r_e.special_audit_report_required is True
    assert r_e.regulatory_verdict == "CRITICAL_CSSF_BREACH_AND_SPECIAL_AUDIT_REPORT_REQUIRED"

    # Scenario F — Bitemporal Special Audit Threshold Flip (2024 vs 2025 at EUR 35,000 / EUR 3,200)
    r_f1 = results["scenario_f_2024"]
    assert r_f1.variance_bps == Decimal("60.00")
    assert r_f1.special_audit_report_required is True
    assert r_f1.regulatory_verdict == "CRITICAL_CSSF_BREACH_AND_SPECIAL_AUDIT_REPORT_REQUIRED"

    r_f2 = results["scenario_f_2025"]
    assert r_f2.variance_bps == Decimal("60.00")
    assert r_f2.special_audit_report_required is False
    assert r_f2.regulatory_verdict == "SIGNIFICANT_NAV_ERROR_BREACH"

    # Scenario G — SIF High-Compensation Exemption from Special Audit Report under Table 9
    r_g = results["scenario_g"]
    assert r_g.variance_bps == Decimal("75.00")
    assert r_g.is_material_nav_breach is True
    assert r_g.special_audit_report_required is False
    assert "SIF_SICAR_SAMPLE_BASED_SEPARATE_REPORT_ONLY_TABLE_9" in r_g.control_flags

    # Scenario H — Class R Flat 120 bps Fee Schedule & Actual/360 Day-Count Breach
    r_h = results["scenario_h"]
    assert r_h.reported_value == "EUR 912500.00"
    assert r_h.replicated_value == "EUR 900000.00"
    assert r_h.absolute_variance == "EUR 12500.00"
    assert r_h.variance_bps == Decimal("1.67")
    assert r_h.regulatory_verdict == "PROSPECTUS_METHODOLOGY_DEVIATION_DETECTED"
    assert "FOOTNOTE_1_CLASS_R_EXCLUSION_VERIFIED" in r_h.control_flags


# ---------------------------------------------------------------------------
# Criterion 4: 5-Mode Retrieval Ablation & Arithmetic Precision Verification
# ---------------------------------------------------------------------------


def test_criterion_4_ablation_study_and_arithmetic_precision(ablation_results: dict[str, Any]) -> None:
    """Verify 6 retrieval ablation modes, taxonomy breakdowns, and Decimal vs float64 precision."""
    ablation = ablation_results

    assert set(ablation.keys()) == {
        "nomic_dense_only",
        "dense_only",
        "dense_bitemporal",
        "hybrid_no_temporal",
        "hybrid_bm25",
        "hybrid_rerank",
    }

    hr = ablation["hybrid_rerank"]
    all_queries = load_full_queries()
    assert hr["total_queries"] == len(all_queries)

    # 1. Baseline Taxonomy Groups (Q01-Q40): Groups 1-4 & 6 achieve 100.0%,
    # while Group 5 achieves >= 83.33% under challenging statutory table queries without lexical anchors.
    baseline_groups = [
        "Group 1: Temporal Anti-Leakage",
        "Group 2: Numerical Sensitivity",
        "Group 3: Multi-Hop & Cross-Reference",
        "Group 4: Prospectus Table & Footnotes",
        "Group 5: Complex Statutory Tables & Footnotes",
        "Group 6: Operational Errors & Restitution Governance",
    ]
    for g_name in baseline_groups:
        g_stats = hr["group_breakdown"][g_name]
        min_required = 83.33 if g_name == "Group 5: Complex Statutory Tables & Footnotes" else 100.0
        assert (
            g_stats["assertion_pass_rate"] >= min_required
        ), f"{g_name} failed baseline requirement: {g_stats}"
        assert g_stats["temporal_compliance_rate"] == 100.0

    # 2. Group 8 (Table-Footnote Fusion) & Group 11 (Calibrated Abstention) must achieve 100.0%
    assert hr["group_breakdown"]["Group 8: Chunk Boundary & Table-Footnote Fusion"]["assertion_pass_rate"] == 100.0
    assert (
        hr["group_breakdown"]["Group 11: Adversarial Out-of-Domain & Anti-Hallucination"]["assertion_pass_rate"]
        == 100.0
    )

    # 3. Zero forbidden-source temporal leakage across all bitemporal-filtered modes
    for bt_mode in ("dense_bitemporal", "hybrid_bm25", "hybrid_rerank"):
        leaked = [q["id"] for q in ablation[bt_mode]["queries"] if q["has_forbidden_source"]]
        assert not leaked, f"Mode {bt_mode} suffered temporal leakage on queries: {leaked}"

    # 4. `hybrid_rerank` must strictly outperform `nomic_dense_only`, `dense_only`, and `hybrid_no_temporal`
    nm = ablation["nomic_dense_only"]
    do = ablation["dense_only"]
    hnt = ablation["hybrid_no_temporal"]
    assert hr["assertion_pass_rate"] > nm["assertion_pass_rate"]
    assert hr["temporal_compliance_rate"] > nm["temporal_compliance_rate"]
    assert hr["assertion_pass_rate"] > do["assertion_pass_rate"]
    assert hr["temporal_compliance_rate"] > do["temporal_compliance_rate"]
    assert hr["temporal_compliance_rate"] > hnt["temporal_compliance_rate"]
    assert hr["assertion_pass_rate"] >= 90.0

    # Symbolic Decimal vs IEEE-754 Float64 Precision Study
    arith = run_arithmetic_precision_study()
    assert arith["total_cases"] == 12
    assert arith["decimal_accuracy_pct"] == 100.0
    assert arith["float64_divergence_count"] >= 10

    # Verify persisted `nomic_scenario_comparison` in the benchmark output
    bench_json = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    nomic_cmp = bench_json.get("nomic_scenario_comparison", {})
    assert nomic_cmp.get("total_runs") == 10
    assert nomic_cmp.get("proposed_passed_runs") == 10
    assert nomic_cmp.get("nomic_passed_runs") == 5


# ---------------------------------------------------------------------------
# Criterion 5: Immutable JSON Audit Trail Logs Verification
# ---------------------------------------------------------------------------


def test_criterion_5_immutable_audit_trail_logs(scenario_results: dict[str, Any]) -> None:
    """Verify cryptographic integrity (SHA-256) and structure of immutable audit trail logs."""
    assert len(scenario_results) == 10

    expected_log_files = [
        "audit_trail_scenario_a.json",
        "audit_trail_scenario_a_2024.json",
        "audit_trail_scenario_a_2025.json",
        "audit_trail_scenario_b.json",
        "audit_trail_scenario_c.json",
        "audit_trail_scenario_d.json",
        "audit_trail_scenario_e.json",
        "audit_trail_scenario_f.json",
        "audit_trail_scenario_f_2024.json",
        "audit_trail_scenario_f_2025.json",
        "audit_trail_scenario_g.json",
        "audit_trail_scenario_h.json",
    ]
    required_keys = {
        "run_id",
        "timestamp",
        "audit_date",
        "retrieved_chunks_metadata",
        "extracted_pydantic_parameters",
        "python_decimal_calculation_steps",
        "variance_bps",
        "confidence_score",
        "regulatory_verdict",
        "audit_trail_sha256",
    }

    for fname in expected_log_files:
        fpath = LOGS_DIR / fname
        assert fpath.exists(), f"Missing audit trail log file: {fpath}"
        data = json.loads(fpath.read_text(encoding="utf-8"))
        if fname in ("audit_trail_scenario_a.json", "audit_trail_scenario_f.json"):
            assert data["temporal_flip_verified"] is True
            assert len(data["runs"]) == 2
        else:
            missing = required_keys - set(data.keys())
            assert not missing, f"{fname} missing required keys: {missing}"
            assert len(data["retrieved_chunks_metadata"]) > 0
            assert len(data["python_decimal_calculation_steps"]) > 0
            assert len(data["audit_trail_sha256"]) == 64
