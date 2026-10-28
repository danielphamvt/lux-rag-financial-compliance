"""Deterministic Decimal verification for NAV audits: fee accrual and NAV pricing replication.

Pure functions shared by the engine graph; no retrieval, no LLM, no state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator

from rag_engine.core.nav_audit_schemas import (
    FundAccountingPackInput,
    ProspectusFeeSpec,
    StatutoryThresholdSpec,
    round_bps,
    round_currency,
)
from rag_engine.core.retrieval_types import BPS_MULTIPLIER


@dataclass(frozen=True)
class VerificationOutcome:
    """Typed return payload from `verify_fee_accrual` and `verify_nav_pricing`."""

    steps: list[dict[str, str]]
    control_flags: list[str]
    verdict: str
    rationale: str
    variance_bps: Decimal
    is_material: bool
    special_audit_req: bool
    reported_val_str: str
    replicated_val_str: str
    abs_var_str: str

    def __iter__(self) -> Iterator[object]:
        """Support 10-tuple unpacking for backward compatibility."""
        return iter((
            self.steps,
            self.control_flags,
            self.verdict,
            self.rationale,
            self.variance_bps,
            self.is_material,
            self.special_audit_req,
            self.reported_val_str,
            self.replicated_val_str,
            self.abs_var_str,
        ))


def compute_replicated_nav_bps(reported_nav: Decimal, replicated_nav: Decimal) -> Decimal:
    """Compute NAV variance in basis points relative to `replicated_nav` (`ROUND_HALF_UP`)."""
    if replicated_nav <= Decimal("0"):
        return Decimal("0.00")
    abs_diff = round_currency(abs(replicated_nav - reported_nav))
    return round_bps((abs_diff / replicated_nav) * BPS_MULTIPLIER)


def _footnote_flag_token(label: str | None) -> str:
    """Derive a `FOOTNOTE_<N>` token from a parsed footnote label (default `FOOTNOTE_1`)."""
    m = re.search(r"(\d+)", label or "")
    return f"FOOTNOTE_{m.group(1)}" if m else "FOOTNOTE_1"


def _section_flag_token(label: str | None) -> str:
    """Derive a `SECTION_<X_Y>` token from a parsed section label (default `SECTION_9_2`)."""
    m = re.search(r"(\d+(?:\.\d+)*)", label or "")
    return f"SECTION_{m.group(1).replace('.', '_')}" if m else "SECTION_9_2"


def _table_flag_token(label: str | None) -> str:
    """Derive a `TABLE_<N>` token from a parsed compensation table label (default `TABLE_9`)."""
    m = re.search(r"Table\s+(\d+)", label or "", flags=re.I)
    return f"TABLE_{m.group(1)}" if m else "TABLE_9"


def verify_fee_accrual(
    inp: FundAccountingPackInput, stat: StatutoryThresholdSpec, prosp: ProspectusFeeSpec
) -> VerificationOutcome:
    """Replicate share-class management fee accrual and compare against administrator-reported fee if supplied."""
    if inp.gav_eur is None or inp.accrual_days is None:
        raise ValueError("gav_eur and accrual_days are required for fee_accrual_audit.")
    gav = round_currency(inp.gav_eur)
    days = Decimal(str(inp.accrual_days))
    denom_correct = Decimal(str(prosp.day_count_denominator))
    if gav <= Decimal("0") or denom_correct <= Decimal("0"):
        raise ValueError("gav_eur and day_count_denominator must be strictly positive.")

    tiered = prosp.has_tiered_reduction
    band_fees: list[Decimal] = []
    tier_steps: list[dict[str, str]] = []
    lower = Decimal("0.00")
    for idx, tier in enumerate(prosp.tiers, start=1):
        upper = tier.threshold_gav_eur
        band_lower = lower
        band = max(Decimal("0.00"), (min(gav, upper) if upper is not None else gav) - band_lower)
        rate = tier.rate_pct / Decimal("100")
        fee_i = round_currency(band * rate * (days / denom_correct))
        band_fees.append(fee_i)
        if tiered:
            cap_lbl = f"EUR {upper:,.0f}" if upper is not None else "no cap"
            bound = f"<= {cap_lbl}" if idx == 1 else f"> EUR {band_lower:,.0f}" + (f" and <= {cap_lbl}" if upper is not None else "")
            fn_lbl = f" {prosp.tier_footnote_label}" if (idx > 1 and prosp.tier_footnote_label) else ""
            tier_steps.append({
                "step": f"{idx}. Tier {idx} GAV Fee Calculation (GAV {bound} @ {tier.rate_pct}%{fn_lbl} {prosp.day_count_convention})",
                "formula": f"{band} * {rate} * ({days} / {denom_correct})",
                "result_eur": str(fee_i),
            })
        elif idx == 1:
            tier_steps.append({
                "step": f"1. Base GAV Fee Calculation ({prosp.share_class} flat @ {tier.rate_pct}% {prosp.day_count_convention})",
                "formula": f"{band} * {rate} * ({days} / {denom_correct})",
                "result_eur": str(fee_i),
            })
        lower = upper if upper is not None else gav

    if not tiered:
        rate = prosp.base_fee_rate_pct / Decimal("100")
        tier_steps.append({
            "step": f"2. Tier 2 GAV Fee Calculation ({prosp.tier_footnote_label or 'Tiered fee reduction'} not applicable to {prosp.share_class})",
            "formula": f"{Decimal('0.00')} * {rate} * ({days} / {denom_correct})",
            "result_eur": "0.00",
        })
        band_fees.append(Decimal("0.00"))

    replicated_fee = round_currency(sum(band_fees, Decimal("0.00")))
    step3 = {
        "step": "3. Total Replicated Annual Management Fee (Python Decimal ROUND_HALF_UP)",
        "formula": " + ".join(str(f) for f in band_fees),
        "result_eur": str(replicated_fee),
    }
    fee_sec_lbl = prosp.fee_schedule_section or "the fee schedule"
    dc_sec_lbl = prosp.day_count_section or "the day-count convention"

    if inp.admin_reported_fee_eur is None:
        rationale = (
            f"Single-class management fee accrual replicated for {prosp.share_class} ({inp.accrual_days} day(s), "
            f"{prosp.day_count_convention} under {fee_sec_lbl} and {dc_sec_lbl}): replicated fee = EUR {replicated_fee}. "
            f"No administrator-reported fee was supplied for variance comparison."
        )
        return VerificationOutcome(
            [*tier_steps, step3], ["SINGLE_CLASS_FEE_REPLICATED"], "REPLICATED_SINGLE_CLASS_FEE_COMPUTED",
            rationale, Decimal("0.00"), False, False, "N/A (not supplied)", f"EUR {replicated_fee}", "EUR 0.00",
        )

    reported_fee = round_currency(inp.admin_reported_fee_eur)
    fee_overstatement = round_currency(reported_fee - replicated_fee)
    abs_fee_variance = round_currency(abs(fee_overstatement))
    fee_error_pct = round_bps((fee_overstatement / replicated_fee) * Decimal("100")) if replicated_fee > Decimal("0") else Decimal("0.00")
    variance_bps = round_bps((abs_fee_variance / gav) * BPS_MULTIPLIER)
    admin_rate_pct = round_currency(
        (inp.admin_used_flat_rate_bps / Decimal("100")) if inp.admin_used_flat_rate_bps is not None else prosp.base_fee_rate_pct
    )
    default_wrong_denom = 360 if prosp.day_count_denominator == 365 else 365
    admin_dc = re.search(r"(?:Actual|30)/(\d{3})", str(inp.admin_used_day_count or ""))
    admin_denom = int(admin_dc.group(1)) if admin_dc else default_wrong_denom
    admin_convention = admin_dc.group(0) if admin_dc else f"Actual/{default_wrong_denom}"
    assumed_lbl = "" if admin_dc else " (assumed; administrator convention not provided)"
    admin_flat_fee = round_currency(gav * (admin_rate_pct / Decimal("100")) * (days / Decimal(admin_denom)))
    step5_formula = (
        f"({reported_fee} - {replicated_fee}) = EUR {fee_overstatement} (+{fee_error_pct}% fee error; {fee_overstatement} / {gav} * 10000 = {variance_bps} bps)"
        if fee_overstatement >= Decimal("0.00")
        else f"|{reported_fee} - {replicated_fee}| = EUR {abs_fee_variance} ({fee_error_pct}% fee error; {abs_fee_variance} / {gav} * 10000 = {variance_bps} bps)"
    )
    steps = [
        *tier_steps,
        step3,
        {
            "step": f"4. Verification of Fund Administrator Erroneous Formula (Flat {admin_rate_pct}% & {admin_convention}{assumed_lbl})",
            "formula": f"{gav} * {admin_rate_pct / Decimal('100')} * ({days} / {admin_denom})",
            "result_eur": str(admin_flat_fee),
        },
        {"step": "5. Fee Accrual Overstatement & Relative NAV Impact in Basis Points (bps)", "formula": step5_formula, "result_bps": str(variance_bps)},
    ]
    control_flags: list[str] = []
    fn_tok = _footnote_flag_token(prosp.tier_footnote_label)
    sec_tok = _section_flag_token(prosp.day_count_section)
    if fee_overstatement == Decimal("0.00"):
        is_material, verdict = False, "COMPLIANT_FEE_ACCRUAL_VERIFIED"
        control_flags.append("PROSPECTUS_FEE_SCHEDULE_COMPLIANT")
        rationale = (
            f"Fund Administrator reported {prosp.share_class} management fees of EUR {reported_fee} match the "
            f"replicated Prospectus fee of EUR {replicated_fee} (0.00 bps variance) under {fee_sec_lbl} and "
            f"{dc_sec_lbl} ({prosp.day_count_convention})."
        )
    else:
        is_material, verdict = variance_bps >= stat.tolerance_threshold_bps, "PROSPECTUS_METHODOLOGY_DEVIATION_DETECTED"
        control_flags.append("INTERNAL_CONTROL_WARNING")
        tier1_cap = prosp.tier_threshold_gav_eur or gav
        if tiered:
            if gav > tier1_cap and reported_fee != replicated_fee:
                control_flags.append("FOOTNOTE_1_TIERED_REDUCTION_OMITTED")
                if fn_tok != "FOOTNOTE_1":
                    control_flags.append(f"{fn_tok}_TIERED_REDUCTION_OMITTED")
        else:
            sc_token = re.sub(r"[^A-Z0-9]+", "_", prosp.share_class.upper()).strip("_")
            control_flags.append(f"{fn_tok}_{sc_token}_EXCLUSION_VERIFIED")
        if admin_denom != prosp.day_count_denominator and (reported_fee == admin_flat_fee or inp.admin_used_day_count is not None):
            control_flags.append(f"{sec_tok}_ACTUAL_{admin_denom}_DENOMINATOR_BREACH")
        if fee_overstatement < Decimal("0.00"):
            rationale = (
                f"Fund Administrator understated {prosp.share_class} management fees by EUR {abs_fee_variance} "
                f"(reported EUR {reported_fee} vs replicated EUR {replicated_fee}, {fee_error_pct}% on fee line, "
                f"{variance_bps} bps on GAV/NAV), deviating from the Prospectus {fee_sec_lbl}/{dc_sec_lbl} methodology."
            )
        elif tiered:
            tier_bps_lbl = str(prosp.tiered_fee_rate_bps or round_bps(prosp.base_fee_rate_pct * Decimal("100")))
            tier_cap_lbl = f"EUR {prosp.tier_threshold_gav_eur:,.0f}" if prosp.tier_threshold_gav_eur is not None else f"EUR {tier1_cap}"
            rationale = (
                f"Fund Administrator overstated {prosp.share_class} management fees by EUR {fee_overstatement} "
                f"(reported EUR {reported_fee} vs replicated EUR {replicated_fee}, +{fee_error_pct}% on fee line, "
                f"{variance_bps} bps on GAV/NAV) due to two Prospectus breaches: (1) ignoring {prosp.tier_footnote_label or 'the tiered reduction'} "
                f"{tier_bps_lbl} bps tiered reduction above {tier_cap_lbl}, and (2) using {admin_convention} instead of {dc_sec_lbl} "
                f"{prosp.day_count_convention}. Although {variance_bps} bps is below the {stat.tolerance_threshold_bps} bps CSSF "
                f"materiality threshold, an INTERNAL_CONTROL_WARNING is triggered to prevent cumulative error buildup."
            )
        else:
            restitution_ref = f"{stat.circular_name} {stat.restitution_ref}" if stat.restitution_ref else stat.circular_name
            rationale = (
                f"Fund Administrator overstated {prosp.share_class} management fees by EUR {fee_overstatement} "
                f"(reported EUR {reported_fee} vs replicated EUR {replicated_fee}, +{fee_error_pct}% on fee line, "
                f"{variance_bps} bps on GAV/NAV). No tiered fee reduction applies to {prosp.share_class} "
                f"(flat {prosp.base_fee_rate_pct}% applies), and using {admin_convention} instead of {dc_sec_lbl} "
                f"{prosp.day_count_convention} caused an unlawful fee overcharge requiring full restitution under {restitution_ref}."
            )
    return VerificationOutcome(
        steps, control_flags, verdict, rationale, variance_bps, is_material,
        False, f"EUR {reported_fee}", f"EUR {replicated_fee}", f"EUR {abs_fee_variance}",
    )


def verify_nav_pricing(
    inp: FundAccountingPackInput, stat: StatutoryThresholdSpec
) -> VerificationOutcome:
    """Replicate NAV per share, variance bps, materiality, and compensation-driven REA report obligations."""
    if inp.replicated_net_assets_eur is None or inp.shares_outstanding is None or inp.reported_nav_per_share is None:
        raise ValueError("replicated_net_assets_eur, shares_outstanding, and reported_nav_per_share are required.")
    net_assets = round_currency(inp.replicated_net_assets_eur)
    shares = inp.shares_outstanding
    reported_nav = round_currency(inp.reported_nav_per_share)
    if shares <= Decimal("0") or reported_nav <= Decimal("0"):
        raise ValueError("shares_outstanding and reported_nav_per_share must be strictly positive.")
    replicated_nav = round_currency(net_assets / shares)
    abs_diff = round_currency(abs(replicated_nav - reported_nav))
    variance_bps = round_bps((abs_diff / reported_nav) * BPS_MULTIPLIER)
    is_material = variance_bps >= stat.tolerance_threshold_bps
    steps = [
        {"step": "1. Replicated NAV per Share (Net Assets / Shares Outstanding, ROUND_HALF_UP)", "formula": f"{net_assets} / {shares}", "result_eur": str(replicated_nav)},
        {"step": "2. Absolute NAV Variance per Share", "formula": f"|{replicated_nav} - {reported_nav}|", "result_eur": str(abs_diff)},
        {"step": "3. Relative NAV Error in Basis Points (bps)", "formula": f"({abs_diff} / {reported_nav}) * 10000", "result_bps": str(variance_bps)},
        {
            "step": f"4. Statutory Materiality Comparison ({stat.circular_name}, Effective {stat.valid_from}..{stat.valid_to})",
            "formula": f"variance_bps ({variance_bps}) vs tolerance_threshold_bps ({stat.tolerance_threshold_bps})",
            "result": "BREACH (>= threshold)" if is_material else "WITHIN_TOLERANCE (< threshold)",
        },
    ]
    control_flags: list[str] = []
    special_audit_req = False
    is_sif_or_sicar = stat.sample_based_separate_report
    tbl_tok = _table_flag_token(stat.compensation_table_label)
    if inp.total_compensation_eur is not None or inp.max_single_investor_eur is not None:
        tot_comp = round_currency(inp.total_compensation_eur or Decimal("0.00"))
        single_comp = round_currency(inp.max_single_investor_eur or Decimal("0.00"))
        exceeds_total = tot_comp > stat.indemnification_total_threshold_eur
        exceeds_single = single_comp > stat.indemnification_single_investor_threshold_eur
        if is_sif_or_sicar:
            control_flags.append(f"SIF_SICAR_SAMPLE_BASED_SEPARATE_REPORT_ONLY_{tbl_tok}")
        else:
            special_audit_req = is_material and (exceeds_total or exceeds_single)
        comp_anchor = (
            f" ({stat.compensation_table_label} {stat.compensation_para_refs})"
            if stat.compensation_table_label and stat.compensation_para_refs
            else ""
        )
        steps.append({
            "step": f"5. Indemnification / Special Audit Report Threshold Check{comp_anchor}",
            "formula": f"entity={inp.fund_type} (SIF/SICAR exempt={is_sif_or_sicar}) | total_compensation (EUR {tot_comp} > EUR {stat.indemnification_total_threshold_eur}: {exceeds_total}) OR single_investor (EUR {single_comp} > EUR {stat.indemnification_single_investor_threshold_eur}: {exceeds_single})",
            "result": "SPECIAL_AUDIT_REPORT_REQUIRED" if special_audit_req else "SEPARATE_SAMPLE_REPORT_ONLY",
        })
    comp_table_lbl = stat.compensation_table_label or "statutory compensation"
    tol_table_lbl = re.search(r"\((Table \d+)\)", stat.fund_category)
    if not is_material:
        verdict = "IMMATERIAL_ERROR_WITHIN_TOLERANCE"
        control_flags.append("WITHIN_STATUTORY_TOLERANCE")
        rationale = f"On audit_date={inp.audit_date}, {stat.circular_name} applies ({stat.fund_category} threshold = {stat.tolerance_threshold_pct}% / {stat.tolerance_threshold_bps} bps). The NAV error of {variance_bps} bps is strictly below {stat.tolerance_threshold_bps} bps; no CSSF notification or retroactive NAV recalculation is required."
    elif special_audit_req:
        verdict = "CRITICAL_CSSF_BREACH_AND_SPECIAL_AUDIT_REPORT_REQUIRED"
        control_flags.extend(["SIGNIFICANT_NAV_CALCULATION_ERROR", "CSSF_IMMEDIATE_NOTIFICATION_REQUIRED", f"REA_SPECIAL_AUDIT_REPORT_REQUIRED_{tbl_tok}"])
        rationale = f"On audit_date={inp.audit_date}, {stat.circular_name} applies ({stat.fund_category} threshold = {stat.tolerance_threshold_pct}% / {stat.tolerance_threshold_bps} bps). The NAV pricing error of {variance_bps} bps exceeds the {stat.tolerance_threshold_bps} bps materiality threshold, and compensation (EUR {inp.total_compensation_eur} total / EUR {inp.max_single_investor_eur} max single investor) exceeds the {comp_table_lbl} thresholds (EUR {stat.indemnification_total_threshold_eur} / EUR {stat.indemnification_single_investor_threshold_eur}), mandating immediate CSSF notification, investor indemnification, and an ad-hoc REA Special Audit Report (Review of all the errors)."
    else:
        verdict = "SIGNIFICANT_NAV_ERROR_BREACH"
        control_flags.extend(["SIGNIFICANT_NAV_CALCULATION_ERROR", "CSSF_IMMEDIATE_NOTIFICATION_REQUIRED"])
        tol_ref = f"under {stat.circular_name} {tol_table_lbl.group(1)}" if tol_table_lbl else f"under {stat.circular_name}"
        rationale = f"On audit_date={inp.audit_date}, {stat.circular_name} applies ({stat.fund_category} threshold = {stat.tolerance_threshold_pct}% / {stat.tolerance_threshold_bps} bps). The NAV error of {variance_bps} bps reaches/exceeds the lowered {stat.tolerance_threshold_bps} bps threshold {tol_ref}, flipping the regulatory verdict to a significant breach requiring CSSF notification and remediation."
    return VerificationOutcome(
        steps, control_flags, verdict, rationale, variance_bps, is_material,
        special_audit_req, f"EUR {reported_nav}/share", f"EUR {replicated_nav}/share", f"EUR {abs_diff}/share",
    )
