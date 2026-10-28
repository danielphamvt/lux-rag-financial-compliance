"""Tool schema contract guardrails & output formatter (`nav_audit_contract_guard.py`).

Validates `run_nav_audit` calls against `FundAccountingPackInput` and formats
`NeuroSymbolicNAVAuditEngine` output with transparent scope and dual-denominator bps.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from domain_profile import PROFILE
from rag_engine.core.nav_audit_schemas import round_bps, round_currency
from rag_engine.core.nav_audit_verifier import compute_replicated_nav_bps


def validate_nav_audit_contract(
    audit_type: str | None,
    share_class: str | None,
    replicated_net_assets_eur: Decimal | None,
    shares_outstanding: Decimal | None,
    reported_nav_per_share: Decimal | None,
    gav_eur: Decimal | None,
) -> str | None:
    """Validate that a `run_nav_audit` call satisfies the `FundAccountingPackInput` schema contract.

    Args:
        audit_type: Requested audit type (`"nav_pricing_error"` or `"fee_accrual_audit"`).
        share_class: Share class identifier for fee accrual audits.
        replicated_net_assets_eur: Replicated net assets in EUR for NAV pricing audits.
        shares_outstanding: Shares outstanding for NAV pricing audits.
        reported_nav_per_share: Administrator-reported NAV per share for NAV pricing audits.
        gav_eur: Gross Asset Value in EUR for fee accrual audits.

    Returns:
        Error message string (`[SCOPE_GUARD: ...]`) if out-of-contract, or `None` if valid.
    """
    supported_audit_types = frozenset(PROFILE.get("supported_audit_types") or ("nav_pricing_error", "fee_accrual_audit"))
    if audit_type and audit_type not in supported_audit_types:
        return (
            f"[SCOPE_GUARD: Unsupported `audit_type={audit_type!r}`. `run_nav_audit` only supports "
            f"{sorted(supported_audit_types)}. Use `calculate_decimal` for other calculations.]"
        )

    supported_classes = frozenset(PROFILE.get("supported_share_classes") or ())
    cls_str = str(share_class or "").strip()
    if cls_str and supported_classes and cls_str not in supported_classes:
        return (
            f"[SCOPE_GUARD: `run_nav_audit` is strictly scoped to a single supported share class per call "
            f"({sorted(supported_classes)}), got {cls_str!r}. For multi-class totals or custom schedules, "
            "call `calculate_decimal` with explicit expressions for each leg, subtotal, and difference.]"
        )

    eff_type = "fee_accrual_audit" if (audit_type == "fee_accrual_audit" or gav_eur is not None) else "nav_pricing_error"
    if eff_type == "nav_pricing_error":
        if (
            replicated_net_assets_eur is None
            or shares_outstanding is None
            or reported_nav_per_share is None
            or shares_outstanding <= Decimal("0")
        ):
            return (
                "[SCOPE_GUARD: `run_nav_audit` (`nav_pricing_error`) requires all three inputs: "
                "`replicated_net_assets_eur`, `shares_outstanding`, and `reported_nav_per_share`. "
                "Use `calculate_decimal` when any of these three inputs is absent.]"
            )
    elif gav_eur is None:
        return (
            "[SCOPE_GUARD: `run_nav_audit` (`fee_accrual_audit`) requires `gav_eur`. "
            "Use `calculate_decimal` for standalone arithmetic.]"
        )
    return None


def _extract_dec(val_str: object) -> Decimal:
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(val_str or ""))
    return Decimal(m.group(0).replace(",", "")) if m else Decimal("0.00")


def format_nav_audit_output(
    res: Any,
    eff_type: str,
    eff_cls: str,
    has_explicit_admin_fee: bool,
    round_bps_fn: Any = round_bps,
    round_currency_fn: Any = round_currency,
) -> str:
    """Format `NeuroSymbolicNAVAuditEngine` output with transparent scope and dual-denominator bps.

    Args:
        res: `NAVAuditVerificationOutput` returned by `NeuroSymbolicNAVAuditEngine.execute_audit`.
        eff_type: Effective audit type (`"nav_pricing_error"` or `"fee_accrual_audit"`).
        eff_cls: Effective share class label.
        has_explicit_admin_fee: Whether `admin_reported_fee_eur` was explicitly supplied.
        round_bps_fn: Basis-point rounding helper (defaults to `round_bps`).
        round_currency_fn: Currency rounding helper (defaults to `round_currency`).

    Returns:
        Formatted audit verification summary string.
    """
    docs_txt = ", ".join(dict.fromkeys(c["source_file"] for c in res.retrieved_chunks_metadata))
    doc_ids_txt = ", ".join(dict.fromkeys(c["doc_id"] for c in res.retrieved_chunks_metadata))

    def _fmt_step_val(s: dict) -> str:
        raw_v = s.get("result_eur") or s.get("result_bps") or s.get("result") or ""
        if s.get("result_eur"):
            try:
                d = Decimal(str(s["result_eur"]).replace(",", ""))
                return f"EUR {d:,.2f} ({d:.2f})"
            except (ValueError, ArithmeticError):
                pass
        return str(raw_v)

    steps_txt = "\n".join(
        f"  - {s['step']}: `{s['formula']}` => **{_fmt_step_val(s)}**" for s in res.python_decimal_calculation_steps
    )
    if eff_type == "fee_accrual_audit" and not has_explicit_admin_fee:
        return (
            f"Document ID: {doc_ids_txt}\n"
            f"File Name: {docs_txt}\n"
            f"Symbolic Audit Verdict: {res.regulatory_verdict} ({eff_cls})\n"
            f"Replicated {eff_cls} Fee: {res.replicated_value} (No admin_reported_fee_eur supplied for variance comparison)\n"
            f"Deterministic Decimal (ROUND_HALF_UP) Steps:\n{steps_txt}\n"
            f"Scope Note: Single-class schedule only ({eff_cls}). For multi-class totals or differences, call `calculate_decimal`."
        )

    if eff_type == "fee_accrual_audit":
        rep_dec = _extract_dec(res.replicated_value)
        adm_dec = _extract_dec(res.reported_value)
        abs_over = round_currency_fn(abs(adm_dec - rep_dec))
        return (
            f"Document ID: {doc_ids_txt}\n"
            f"File Name: {docs_txt}\n"
            f"Symbolic Audit Verdict: {res.regulatory_verdict}\n"
            f"Reported vs Replicated ({eff_cls} only): EUR {adm_dec:,.2f} vs EUR {rep_dec:,.2f} "
            f"(Absolute Variance: EUR {abs_over:,.2f} [{abs_over:.2f}], Single-Class GAV Impact: {res.variance_bps} bps vs Relative Bps Threshold: {res.applicable_threshold_bps} bps)\n"
            f"Single-Class Relative Bps Materiality (>= {res.applicable_threshold_bps} bps): {res.is_material_nav_breach} | "
            f"Bps-Based REA Flag: {res.special_audit_report_required} "
            f"(Note: engine boolean flags evaluate single-class relative bps only; compare Absolute Variance EUR {abs_over:,.2f} against any statutory absolute EUR reporting thresholds retrieved from the corpus separately)\n"
            f"Control Flags: {', '.join(res.control_flags)}\n"
            f"Deterministic Decimal (ROUND_HALF_UP) Steps:\n{steps_txt}\n"
            f"Rationale: {res.verdict_rationale}"
        )

    reported_nav_dec = _extract_dec(res.reported_value)
    rep_nav_dec = _extract_dec(res.replicated_value)
    abs_var_dec = _extract_dec(res.absolute_variance)
    rep_base_bps = compute_replicated_nav_bps(reported_nav_dec, rep_nav_dec) if rep_nav_dec else res.variance_bps
    rep_base_pct = round_bps_fn((abs_var_dec / rep_nav_dec) * Decimal("100")) if rep_nav_dec else Decimal("0.00")
    if rep_nav_dec and str(rep_base_bps) != str(res.variance_bps):
        step3b = (
            f"  - 3b. Relative NAV Error in Basis Points (on Replicated NAV Base): "
            f"`({abs_var_dec:.2f} / {rep_nav_dec:.2f}) * 10000` => **{rep_base_bps} bps** (**{rep_base_pct}%**)"
        )
        steps_txt = f"{steps_txt}\n{step3b}"
    return (
        f"Document ID: {doc_ids_txt}\n"
        f"File Name: {docs_txt}\n"
        f"Symbolic Audit Verdict: {res.regulatory_verdict}\n"
        f"Reported vs Replicated: {res.reported_value} vs {res.replicated_value} "
        f"(Absolute Variance: {res.absolute_variance}, Relative Variance: {res.variance_bps} bps on reported NAV base / "
        f"{rep_base_bps} bps ({rep_base_pct}%) on replicated NAV base vs Threshold: {res.applicable_threshold_bps} bps)\n"
        f"Material Breach: {res.is_material_nav_breach} | REA Special Audit Report Required: {res.special_audit_report_required}\n"
        f"Control Flags: {', '.join(res.control_flags)}\n"
        f"Deterministic Decimal (ROUND_HALF_UP) Steps:\n{steps_txt}\n"
        f"Rationale: {res.verdict_rationale}"
    )
