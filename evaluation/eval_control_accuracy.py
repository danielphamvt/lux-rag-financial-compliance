"""Automated PyTest Suite for Second-Level NAV Control Accuracy & Confusion Matrix.

Verifies the 30-case synthetic control dataset (`evaluation/eval_dataset_control_cases.json`),
Confusion Matrix safety metrics (TP=10, FP=0, TN=20, FN=0, Recall=100%, FPR=0%, FNR=0%),
and the 3-pillar Audit Report Quality Rubric via `evaluation/evaluate_control_accuracy.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from evaluation.evaluate_control_accuracy import (
    CONTROL_BENCHMARK_OUTPUT_PATH,
    CONTROL_LOGS_DIR,
    load_control_cases,
    run_control_accuracy_benchmark,
)
from rag_engine.core.nav_audit import NeuroSymbolicNAVAuditEngine
from rag_engine.core.registry import get_backend


@pytest.fixture(scope="module")
def audit_engine() -> NeuroSymbolicNAVAuditEngine:
    """Provide a fast deterministic NeuroSymbolicNAVAuditEngine backed by `page_index`."""
    backend = get_backend("page_index")
    return NeuroSymbolicNAVAuditEngine(
        retriever=backend.retriever,
        logs_dir=CONTROL_LOGS_DIR,
    )


def test_control_dataset_structure_and_distribution() -> None:
    """Verify the 30-case synthetic control dataset distribution and schema keys."""
    cases = load_control_cases()
    assert len(cases) == 30

    required_keys = {
        "case_id",
        "case_name",
        "audit_date",
        "fund_category",
        "share_class",
        "gav_eur",
        "admin_nav_per_share",
        "replicated_nav_per_share",
        "shares_outstanding",
        "admin_fee_eur",
        "expected_fee_eur",
        "ground_truth_verdict",
        "ground_truth_has_breach",
        "expected_statutory_ref",
    }
    for case in cases:
        missing = required_keys - set(case.keys())
        assert not missing, f"Case {case.get('case_id')} missing keys: {missing}"

    clean_cases = [c for c in cases if c.get("case_category") == "clean_nav_run"]
    pricing_cases = [c for c in cases if c.get("case_category") == "pricing_breach"]
    accrual_cases = [c for c in cases if c.get("case_category") == "accrual_breach"]
    acceptable_cases = [c for c in cases if c.get("case_category") == "acceptable_variance"]
    canonical_scenarios = [c["scenario_id"] for c in cases if c.get("scenario_id")]

    assert len(clean_cases) == 15
    assert len(pricing_cases) == 7
    assert len(accrual_cases) == 3
    assert len(acceptable_cases) == 5
    assert len(canonical_scenarios) == 10
    assert set(canonical_scenarios) == {
        "scenario_a_2024",
        "scenario_a_2025",
        "scenario_b",
        "scenario_c",
        "scenario_d",
        "scenario_e",
        "scenario_f_2024",
        "scenario_f_2025",
        "scenario_g",
        "scenario_h",
    }

    # Verify both CSSF 02/77 (2024) and CSSF 24/856 (2025) regimes are represented
    has_2024 = any(str(c["audit_date"]).startswith("2024-") for c in cases)
    has_2025 = any(str(c["audit_date"]).startswith("2025-") for c in cases)
    assert has_2024 and has_2025


def test_control_accuracy_metrics(audit_engine: NeuroSymbolicNAVAuditEngine) -> None:
    """Verify Confusion Matrix metrics (EDR, FPR, FNR) and Audit Report Quality Rubric."""
    metrics: dict[str, Any] = run_control_accuracy_benchmark(engine=audit_engine)

    assert metrics["error_detection_rate"] >= 0.95
    assert metrics["false_positive_rate"] <= 0.05
    assert metrics["false_negative_rate"] == 0.0

    assert metrics["total_runs"] == 30
    assert metrics["clean_runs"] == 20
    assert metrics["pure_clean_runs"] == 15
    assert metrics["acceptable_noise_runs"] == 5
    assert metrics["breach_injected_runs"] == 10
    assert metrics["confusion_matrix"] == {"tp": 10, "fp": 0, "tn": 20, "fn": 0}
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1_score"] == 1.0

    assert metrics["statutory_grounding_score"] == 1.0
    assert metrics["mathematical_traceability_score"] == 1.0
    assert metrics["cryptographic_traceability_score"] == 1.0
    assert metrics["audit_report_quality_score"] == 1.0

    assert CONTROL_BENCHMARK_OUTPUT_PATH.exists()
    persisted = json.loads(CONTROL_BENCHMARK_OUTPUT_PATH.read_text(encoding="utf-8"))
    assert persisted["confusion_matrix"] == {"tp": 10, "fp": 0, "tn": 20, "fn": 0}
    assert len(persisted["cases"]) == 30
