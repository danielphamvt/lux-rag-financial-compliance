"""Pydantic Schemas and Decimal Rounding Helpers for NAV Audit.

Shared across retrieval backends and the frontend LangGraph agent via `rag_engine.core.nav_audit`.
Semantic rule parsers live in `rag_engine.core.nav_audit_parsers`.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


TWO_DECIMALS: Decimal = Decimal("0.01")


def round_currency(val: Decimal) -> Decimal:
    """Standard commercial accounting rounding to 2 decimal places (`ROUND_HALF_UP`)."""
    return val.quantize(TWO_DECIMALS, rounding=ROUND_HALF_UP)


round_bps = round_currency


class StatutoryThresholdSpec(BaseModel):
    """Pydantic schema for statutory NAV error tolerance and indemnification rules."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    doc_id: str = Field(description="Source regulatory circular ID (e.g. CSSF_02_77 or CSSF_24_856)")
    circular_name: str = Field(description="Human-readable circular name")
    valid_from: str = Field(description="Effective start date (YYYY-MM-DD)")
    valid_to: str = Field(description="Effective end date (YYYY-MM-DD)")
    fund_category: str = Field(description="Matched UCI category from statutory table")
    tolerance_threshold_pct: Decimal = Field(description="Statutory tolerance threshold in % of NAV")
    tolerance_threshold_bps: Decimal = Field(description="Statutory tolerance threshold in basis points (bps)")
    indemnification_total_threshold_eur: Decimal = Field(description="Total fund compensation threshold in EUR")
    indemnification_single_investor_threshold_eur: Decimal = Field(description="Single investor compensation threshold in EUR")
    special_report_rule: str = Field(description="Statutory rule governing External Auditor (REA) Special Report")
    sample_based_separate_report: bool = Field(
        default=False,
        description="Entity matches the SIF/SICAR vocabulary and the retrieved summary tables route significant errors to sample-based separate reports only",
    )
    compensation_table_label: str = Field(default="", description="Parsed label of the significant-error summary table (e.g. Table 9); empty when absent from retrieved text")
    compensation_para_refs: str = Field(default="", description="Parsed paragraph references for the compensation thresholds and summary table (e.g. Para 138/149); empty when absent")
    restitution_ref: str = Field(default="", description="Parsed section/point citation governing overcharged-fee restitution; empty when absent")
    source_citation: str = Field(description="Exact page/table/paragraph citation from retrieved chunks")


class FeeTierSpec(BaseModel):
    """One band of a share-class fee schedule: `rate_pct` applies to the GAV portion up to `threshold_gav_eur` (None = unbounded top band)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    threshold_gav_eur: Decimal | None = Field(default=None, description="Upper GAV bound of this band in EUR; None = unbounded top band")
    rate_pct: Decimal = Field(description="Annual management fee rate in % applying to the GAV portion within this band")
    rate_bps: Decimal = Field(description="Annual management fee rate in bps")


class ProspectusFeeSpec(BaseModel):
    """Pydantic schema for share-class fee schedule, tiered footnotes, and day-count convention."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    doc_id: str = Field(description="Prospectus document ID (explicit input or corpus-registry resolution)")
    fund_name: str = Field(description="Fund name the fee schedule was parsed for")
    share_class: str = Field(description="Target share class (e.g. Class I, Class R, Class H-USD)")
    tiers: list[FeeTierSpec] = Field(description="Ordered fee-schedule bands; two or more bands = tiered reduction")
    fee_schedule_section: str = Field(default="", description="Parsed heading anchor of the fee schedule (e.g. Section 9.1); empty when absent")
    day_count_section: str = Field(default="", description="Parsed heading anchor of the day-count convention (e.g. Section 9.2); empty when absent")
    tier_footnote_label: str | None = Field(default=None, description="Parsed label of the tiered-reduction footnote (e.g. Footnote (*1))")
    day_count_convention: str = Field(description="Mandated day-count convention (e.g. Actual/365, Actual/360, 30/360)")
    day_count_denominator: int = Field(description="Denominator days in year (e.g. 365 or 360)")
    source_citation: str = Field(description="Exact section & footnote citation derived from retrieved chunks")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_fee_rate_pct(self) -> Decimal:
        """Base (first-band) annual management fee rate in %."""
        return self.tiers[0].rate_pct

    @computed_field  # type: ignore[prop-decorator]
    @property
    def base_fee_rate_bps(self) -> Decimal:
        """Base (first-band) annual management fee rate in bps."""
        return self.tiers[0].rate_bps

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_tiered_reduction(self) -> bool:
        """Whether the parsed schedule applies a tiered reduction above the first band."""
        return len(self.tiers) > 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tier_threshold_gav_eur(self) -> Decimal | None:
        """GAV boundary where the reduced band starts (first band's cap) in EUR, or None when untiered."""
        return self.tiers[0].threshold_gav_eur if len(self.tiers) > 1 else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tiered_fee_rate_pct(self) -> Decimal | None:
        """Reduced (second-band) annual management fee rate in %, or None when untiered."""
        return self.tiers[1].rate_pct if len(self.tiers) > 1 else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tiered_fee_rate_bps(self) -> Decimal | None:
        """Reduced (second-band) annual management fee rate in bps, or None when untiered."""
        return self.tiers[1].rate_bps if len(self.tiers) > 1 else None


class FundAccountingPackInput(BaseModel):
    """Input payload from First-Level Fund Administrator to be verified by Second-Level Audit Engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    scenario_id: str = Field(description="Scenario or run identifier")
    audit_date: str = Field(description="Valuation / audit date in YYYY-MM-DD format")
    audit_type: Literal["nav_pricing_error", "fee_accrual_audit"] = Field(default="nav_pricing_error")
    fund_name: str = Field(description="Name of the UCI / SICAV")
    fund_type: str = Field(description="UCI category (e.g. Money Market UCI / MMF, Bond UCITS, Equity UCITS)")
    share_class: str | None = Field(default=None)
    prospectus_doc_id: str | None = Field(default=None, description="Explicit prospectus document ID; when None the fund name is resolved against the corpus registry")
    replicated_net_assets_eur: Decimal | None = Field(default=None)
    shares_outstanding: Decimal | None = Field(default=None)
    reported_nav_per_share: Decimal | None = Field(default=None)
    total_compensation_eur: Decimal | None = Field(default=None)
    max_single_investor_eur: Decimal | None = Field(default=None)
    gav_eur: Decimal | None = Field(default=None)
    accrual_days: int | None = Field(default=None)
    admin_reported_fee_eur: Decimal | None = Field(default=None, description="Administrator's reported management fee in EUR (for fee_accrual_audit); None for replication-only.")
    admin_used_day_count: str | None = Field(default=None)
    admin_used_flat_rate_bps: Decimal | None = Field(default=None)


class NAVAuditResult(BaseModel):
    """Immutable Neuro-Symbolic Audit Result & Regulatory Verdict."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    run_id: str
    scenario_id: str
    timestamp: str
    audit_date: str
    fund_name: str
    fund_type: str
    share_class: str | None = None
    retrieved_chunks_metadata: list[dict[str, Any]]
    extracted_pydantic_parameters: dict[str, Any]
    python_decimal_calculation_steps: list[dict[str, str]]
    reported_value: str
    replicated_value: str
    absolute_variance: str
    variance_bps: Decimal
    applicable_threshold_bps: Decimal
    is_material_nav_breach: bool
    special_audit_report_required: bool
    confidence_score: float
    control_flags: list[str]
    regulatory_verdict: str
    verdict_rationale: str
    prospectus_resolved_via: str = ""
    audit_trail_path: str = ""
    audit_trail_sha256: str = ""
