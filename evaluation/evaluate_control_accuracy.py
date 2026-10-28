"""Empirical Control Accuracy & Confusion Matrix Evaluation Runner.

Evaluates the Neuro-Symbolic Second-Level NAV Control Engine across 30 synthetic
fund accounting packs (15 Clean NAV Runs, 5 Pricing Breaches, 5 Accrual &
Prospectus Breaches, 5 Acceptable Variances) under Circular CSSF 02/77 and
Circular CSSF 24/856, computing Confusion Matrix metrics (TP, FP, TN, FN,
Recall/EDR, FPR, FNR, Precision, F1-Score) and the 3-pillar Audit Report
Quality Rubric.
"""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

# Strictly enforce isolated test environment (.env.test) before loading any modules
from evaluation.eval_common import ensure_runtime

ensure_runtime()

from rag_engine.core.nav_audit import NeuroSymbolicNAVAuditEngine, _write_json_atomic
from rag_engine.core.nav_audit_schemas import (
    FundAccountingPackInput,
    NAVAuditResult,
    round_currency,
)
from rag_engine.core.registry import get_backend

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONTROL_CASES_DATASET_PATH = WORKSPACE_ROOT / "evaluation" / "eval_dataset_control_cases.json"
CONTROL_BENCHMARK_OUTPUT_PATH = (
    WORKSPACE_ROOT / "evaluation" / "results" / "control_accuracy_benchmark.json"
)
CONTROL_LOGS_DIR = WORKSPACE_ROOT / "logs" / "control_cases"
DEFAULT_CONTROL_BACKEND = "page_index"
DEFAULT_ACCRUAL_DAYS = 365
DEFAULT_FUND_NAME = "Lux Alpha SICAV"
SUB_2BPS_CLEAN_THRESHOLD = Decimal("2.00")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def load_control_cases(dataset_path: Path | str = CONTROL_CASES_DATASET_PATH) -> list[dict[str, Any]]:
    """Load the 30 synthetic NAV control benchmark cases from JSON.

    Args:
        dataset_path: Path to `eval_dataset_control_cases.json`.

    Returns:
        List of 30 control case specification dictionaries.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Control cases dataset not found at: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("cases"), list):
        return raw["cases"]
    raise ValueError(f"Invalid control dataset format in {path}: expected list of case objects.")


def build_accounting_pack_input(
    case: dict[str, Any],
    use_canonical_id: bool = False,
) -> FundAccountingPackInput:
    """Convert a unified control/audit case dictionary into `FundAccountingPackInput`.

    Args:
        case: Single control case dictionary from `eval_dataset_control_cases.json`.
        use_canonical_id: If True and `scenario_id` is present on `case`, use `scenario_id`
            (e.g. `scenario_a_2024`) instead of `case_id` (`CTRL_26`).

    Returns:
        Validated `FundAccountingPackInput` ready for `NeuroSymbolicNAVAuditEngine`.
    """
    admin_nav = Decimal(str(case["admin_nav_per_share"]))
    replicated_nav = Decimal(str(case["replicated_nav_per_share"]))
    shares = Decimal(str(case["shares_outstanding"]))
    gav = Decimal(str(case["gav_eur"]))
    admin_fee = Decimal(str(case["admin_fee_eur"]))

    has_fee_accrual_params = any(
        case.get(k) is not None
        for k in ("accrual_days", "admin_used_day_count", "admin_used_flat_rate_bps")
    )
    audit_type = str(
        case.get("audit_type")
        or ("fee_accrual_audit" if has_fee_accrual_params else "nav_pricing_error")
    )
    run_scenario_id = (
        str(case["scenario_id"])
        if use_canonical_id and case.get("scenario_id")
        else str(case["case_id"])
    )
    tot_comp = (
        Decimal(str(case["total_compensation_eur"]))
        if case.get("total_compensation_eur") is not None
        else None
    )
    single_comp = (
        Decimal(str(case["max_single_investor_eur"]))
        if case.get("max_single_investor_eur") is not None
        else None
    )
    flat_bps = (
        Decimal(str(case["admin_used_flat_rate_bps"]))
        if case.get("admin_used_flat_rate_bps") is not None
        else None
    )

    return FundAccountingPackInput(
        scenario_id=run_scenario_id,
        audit_date=str(case["audit_date"]),
        audit_type=audit_type,
        fund_name=str(case.get("fund_name", DEFAULT_FUND_NAME)),
        fund_type=str(case["fund_category"]),
        share_class=case.get("share_class"),
        reported_net_assets_eur=round_currency(admin_nav * shares),
        replicated_net_assets_eur=round_currency(replicated_nav * shares),
        shares_outstanding=shares,
        reported_nav_per_share=admin_nav,
        total_compensation_eur=tot_comp,
        max_single_investor_eur=single_comp,
        gav_eur=gav,
        accrual_days=int(case.get("accrual_days", DEFAULT_ACCRUAL_DAYS)),
        admin_reported_fee_eur=admin_fee,
        admin_used_day_count=case.get("admin_used_day_count"),
        admin_used_flat_rate_bps=flat_bps,
    )


def detect_system_breach(result: NAVAuditResult) -> bool:
    """Determine whether the Neuro-Symbolic Audit Engine flagged a compliance breach.

    Args:
        result: `NAVAuditResult` returned by `NeuroSymbolicNAVAuditEngine.execute_audit`.

    Returns:
        True if a material NAV pricing breach or Prospectus fee methodology deviation was flagged.
    """
    return bool(
        result.is_material_nav_breach
        or result.regulatory_verdict == "PROSPECTUS_METHODOLOGY_DEVIATION_DETECTED"
        or "INTERNAL_CONTROL_WARNING" in result.control_flags
    )


def evaluate_audit_report_rubric(case: dict[str, Any], result: NAVAuditResult) -> dict[str, Any]:
    """Score a single `NAVAuditResult` against the 3-pillar Audit Report Quality Rubric.

    Rubric criteria:
    1. Statutory Grounding Coverage: Accurate bitemporal circular (`CSSF 02/77` vs `CSSF 24/856`),
       statutory table/section citation, and exact bps tolerance threshold extraction.
    2. Step-Level Mathematical Traceability: Complete `python_decimal_calculation_steps`
       covering NAV/GAV, tiered fees, and basis-point variance matching `result.variance_bps`.
    3. Cryptographic Auditability: Every retrieved chunk carries a 64-char SHA-256 signature
       and the audit report carries a verified 64-char `audit_trail_sha256` signature.

    Args:
        case: Ground-truth control case dictionary.
        result: Executed `NAVAuditResult`.

    Returns:
        Dictionary with boolean checks and numeric rubric scores in `[0.0, 1.0]`.
    """
    stat_spec = result.extracted_pydantic_parameters.get("statutory_threshold_spec", {})
    expected_ref = str(case.get("expected_statutory_ref", ""))
    expected_circular = "CSSF_02_77" if str(case["audit_date"]) <= "2024-12-31" else "CSSF_24_856"
    citation = str(stat_spec.get("source_citation", ""))
    chunks = result.retrieved_chunks_metadata
    chunk_doc_ids = {str(c.get("doc_id", "")) for c in chunks}
    chunk_sections_and_excerpts = " ".join(
        f"{c.get('section_id', '')} {c.get('excerpt', '')}" for c in chunks
    )

    bps_match = re.search(r"\((\d+)\s*bps\)", expected_ref)
    expected_bps = Decimal(bps_match.group(1)) if bps_match else result.applicable_threshold_bps
    clause_tokens = [tok for tok in ("Table 1", "Table 2", "Section I.2") if tok in expected_ref]
    clause_matched = all(
        tok in citation and tok in chunk_sections_and_excerpts for tok in clause_tokens
    )

    prosp_ok = True
    if "Prospectus" in expected_ref:
        prosp_spec = result.extracted_pydantic_parameters.get("prospectus_fee_spec", {})
        prosp_cit = str(prosp_spec.get("source_citation", ""))
        prosp_ok = (
            "LUX_ALPHA_PROSPECTUS" in chunk_doc_ids
            and "Section 9.1" in prosp_cit
            and "Section 9.2" in prosp_cit
        )

    statutory_grounding_ok = bool(
        stat_spec.get("doc_id") == expected_circular
        and expected_circular in chunk_doc_ids
        and Decimal(str(stat_spec.get("tolerance_threshold_bps", "-1"))) == expected_bps
        and clause_matched
        and prosp_ok
    )

    steps = result.python_decimal_calculation_steps
    min_steps = 5 if "Prospectus" in expected_ref else 4
    steps_have_formulas = len(steps) >= min_steps and all(
        bool(s.get("step")) and bool(s.get("formula")) for s in steps
    )
    bps_step_matches = any(s.get("result_bps") == str(result.variance_bps) for s in steps)
    if "Prospectus" in expected_ref and case.get("expected_fee_eur") is not None:
        expected_rep_str = f"EUR {round_currency(Decimal(str(case['expected_fee_eur'])))}"
    else:
        expected_rep_str = f"EUR {round_currency(Decimal(str(case['replicated_nav_per_share'])))}/share"
    replicated_val_matches = result.replicated_value == expected_rep_str
    special_report_matches = (
        result.special_audit_report_required == bool(case["expected_special_report"])
        if "expected_special_report" in case
        else True
    )
    math_traceability_ok = bool(
        steps_have_formulas
        and bps_step_matches
        and replicated_val_matches
        and special_report_matches
        and bool(result.verdict_rationale.strip())
    )
    chunks_have_sha256 = len(chunks) > 0 and all(
        bool(SHA256_HEX_RE.match(str(c.get("sha256", "")))) for c in chunks
    )
    report_has_sha256 = bool(SHA256_HEX_RE.match(str(result.audit_trail_sha256)))
    trail_exists = bool(result.audit_trail_path) and Path(result.audit_trail_path).exists()
    crypto_auditability_ok = bool(chunks_have_sha256 and report_has_sha256 and trail_exists)

    s_score = 1.0 if statutory_grounding_ok else 0.0
    m_score = 1.0 if math_traceability_ok else 0.0
    c_score = 1.0 if crypto_auditability_ok else 0.0
    return {
        "statutory_grounding_ok": statutory_grounding_ok,
        "mathematical_traceability_ok": math_traceability_ok,
        "cryptographic_auditability_ok": crypto_auditability_ok,
        "statutory_grounding_score": s_score,
        "mathematical_traceability_score": m_score,
        "cryptographic_traceability_score": c_score,
        "overall_rubric_score": round((s_score + m_score + c_score) / 3.0, 4),
    }


def format_confusion_matrix_banner(metrics: dict[str, Any]) -> str:
    """Format the terminal ASCII report for the Control Accuracy & Confusion Matrix Benchmark.

    Args:
        metrics: Aggregated benchmark metrics dictionary.

    Returns:
        Formatted multi-line ASCII table string.
    """
    cm = metrics["confusion_matrix"]
    edr_pct = metrics["error_detection_rate"] * 100.0
    fpr_pct = metrics["false_positive_rate"] * 100.0
    fnr_pct = metrics["false_negative_rate"] * 100.0
    acc_pct = metrics["accuracy"] * 100.0
    stat_pct = metrics["statutory_grounding_score"] * 100.0
    cryp_pct = metrics["cryptographic_traceability_score"] * 100.0

    lines = [
        "=" * 88,
        "SECOND-LEVEL NAV CONTROL ACCURACY & CONFUSION MATRIX BENCHMARK",
        "=" * 88,
        f"Total Evaluated Runs: {metrics['total_runs']}",
        (
            f"Clean Runs: {metrics['clean_runs']} "
            f"({metrics['pure_clean_runs']} Pure Clean + {metrics['acceptable_noise_runs']} Acceptable Noise) "
            f"| Breach Injected Runs: {metrics['breach_injected_runs']}"
        ),
        "",
        "CONFUSION MATRIX:",
        f"  True Positives (TP):  {cm['tp']:2d} | False Positives (FP): {cm['fp']}",
        f"  True Negatives (TN):  {cm['tn']:2d} | False Negatives (FN): {cm['fn']}",
        "",
        "KEY SAFETY METRICS:",
        f"  - Error Detection Rate (Sensitivity / Recall): {edr_pct:5.1f}%",
        f"  - False Positive Rate (FPR / Alert Fatigue):   {fpr_pct:5.1f}%",
        f"  - False Negative Rate (FNR / Undetected Risk): {fnr_pct:5.1f}%",
        f"  - Overall Classification Accuracy:             {acc_pct:5.1f}%",
        f"  - Audit Report Statutory Grounding Score:      {stat_pct:5.1f}%",
        f"  - Cryptographic Traceability Verification:     {cryp_pct:5.1f}%",
        "=" * 88,
    ]
    return "\n".join(lines)


def run_control_accuracy_benchmark(
    engine: NeuroSymbolicNAVAuditEngine | None = None,
    dataset_path: Path | str = CONTROL_CASES_DATASET_PATH,
    output_path: Path | str = CONTROL_BENCHMARK_OUTPUT_PATH,
    backend_name: str = DEFAULT_CONTROL_BACKEND,
    print_report: bool = True,
) -> dict[str, Any]:
    """Execute the 30-case Second-Level NAV Control Accuracy & Confusion Matrix benchmark.

    Args:
        engine: Optional pre-initialized `NeuroSymbolicNAVAuditEngine`. When omitted,
            initializes an engine backed by `get_backend(backend_name).retriever`.
        dataset_path: Path to `eval_dataset_control_cases.json`.
        output_path: Destination JSON path for persisted benchmark results.
        backend_name: Backend identifier used when `engine` is None (defaults to `page_index`).
        print_report: Whether to print the formatted ASCII table to stdout.

    Returns:
        Dictionary containing confusion matrix counts, safety rates (`error_detection_rate`,
        `false_positive_rate`, `false_negative_rate`, `accuracy`, `precision`, `f1_score`),
        rubric scores, and per-case outcomes.
    """
    if engine is None:
        backend = get_backend(backend_name)
        engine = NeuroSymbolicNAVAuditEngine(
            retriever=backend.retriever,
            logs_dir=CONTROL_LOGS_DIR,
        )

    cases = load_control_cases(dataset_path)
    tp = fp = tn = fn = 0
    pure_clean_runs = acceptable_noise_runs = breach_injected_runs = 0
    case_reports: list[dict[str, Any]] = []

    for case in cases:
        inp = build_accounting_pack_input(case)
        result = engine.execute_audit(inp)
        predicted_breach = detect_system_breach(result)
        gt_breach = bool(case["ground_truth_has_breach"])

        if gt_breach:
            breach_injected_runs += 1
            if predicted_breach:
                tp += 1
                cm_label = "TP"
            else:
                fn += 1
                cm_label = "FN"
        else:
            if (
                case.get("case_category") == "acceptable_variance"
                or case.get("ground_truth_verdict") == "ACCEPTABLE_VARIANCE_WITH_NOTE"
                or result.variance_bps >= SUB_2BPS_CLEAN_THRESHOLD
            ):
                acceptable_noise_runs += 1
            else:
                pure_clean_runs += 1
            if predicted_breach:
                fp += 1
                cm_label = "FP"
            else:
                tn += 1
                cm_label = "TN"

        rubric = evaluate_audit_report_rubric(case, result)
        case_reports.append({
            "case_id": case["case_id"],
            "case_name": case["case_name"],
            "case_category": case.get("case_category", ""),
            "audit_date": case["audit_date"],
            "fund_category": case["fund_category"],
            "share_class": case.get("share_class"),
            "ground_truth_verdict": case["ground_truth_verdict"],
            "ground_truth_has_breach": gt_breach,
            "predicted_has_breach": predicted_breach,
            "confusion_matrix_class": cm_label,
            "regulatory_verdict": result.regulatory_verdict,
            "variance_bps": str(result.variance_bps),
            "applicable_threshold_bps": str(result.applicable_threshold_bps),
            "control_flags": result.control_flags,
            "audit_trail_sha256": result.audit_trail_sha256,
            "rubric": rubric,
        })

    total = len(cases)
    pos_total = tp + fn
    neg_total = fp + tn
    edr = (tp / pos_total) if pos_total > 0 else 0.0
    fpr = (fp / neg_total) if neg_total > 0 else 0.0
    fnr = (fn / pos_total) if pos_total > 0 else 0.0
    acc = ((tp + tn) / total) if total > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * precision * edr / (precision + edr)) if (precision + edr) > 0 else 0.0

    stat_score = sum(c["rubric"]["statutory_grounding_score"] for c in case_reports) / total if total else 0.0
    math_score = sum(c["rubric"]["mathematical_traceability_score"] for c in case_reports) / total if total else 0.0
    cryp_score = sum(c["rubric"]["cryptographic_traceability_score"] for c in case_reports) / total if total else 0.0
    rubric_avg = round((stat_score + math_score + cryp_score) / 3.0, 4)

    metrics: dict[str, Any] = {
        "total_runs": total,
        "clean_runs": pure_clean_runs + acceptable_noise_runs,
        "pure_clean_runs": pure_clean_runs,
        "acceptable_noise_runs": acceptable_noise_runs,
        "breach_injected_runs": breach_injected_runs,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "error_detection_rate": round(edr, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "statutory_grounding_score": round(stat_score, 4),
        "mathematical_traceability_score": round(math_score, 4),
        "cryptographic_traceability_score": round(cryp_score, 4),
        "audit_report_quality_score": rubric_avg,
        "cases": case_reports,
    }

    _write_json_atomic(Path(output_path), metrics)
    if print_report:
        print(format_confusion_matrix_banner(metrics))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Second-Level NAV Control Confusion Matrix Benchmark.")
    parser.add_argument("--backend", default=DEFAULT_CONTROL_BACKEND, help="Backend name (default: page_index)")
    args = parser.parse_args()
    run_control_accuracy_benchmark(backend_name=args.backend, print_report=True)
