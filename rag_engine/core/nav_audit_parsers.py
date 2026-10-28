"""Semantic Rule Parsers for NAV Audit: statutory thresholds and prospectus fee schedules.

Extracts structured Pydantic specs from retrieved parent chunks; shared by the
retrieval backends and the frontend LangGraph agent via `rag_engine.core.nav_audit`.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from rag_engine.core.nav_audit_schemas import (
    FeeTierSpec,
    ProspectusFeeSpec,
    StatutoryThresholdSpec,
    round_bps,
)
from rag_engine.core.retrieval_types import RetrievalResponse


def _require_match(pattern: str, text: str, field_desc: str, flags: int = re.I) -> re.Match[str]:
    """Require a regex match on retrieved text or raise a descriptive `ValueError`."""
    match = re.search(pattern, text, flags)
    if match is None:
        raise ValueError(f"Failed to extract required parameter '{field_desc}' from retrieved chunks.")
    return match


def _build_statutory_citation_from_chunks(
    chunks: list[Any], doc_id: str, circular_name: str
) -> str:
    """Construct `source_citation` dynamically from the actual retrieved parent chunks."""
    doc_chunks = [c for c in chunks if getattr(c, "doc_id", "") == doc_id]
    refs: list[str] = []
    for ch in doc_chunks:
        sec = str(getattr(ch, "section_id", "") or "").strip()
        pg = getattr(ch, "page_number", None)
        txt = str(getattr(ch, "page_content", "") or "")
        loc = f"{sec} (Page {pg})" if (sec and pg is not None and f"page {pg}" not in sec.lower()) else (sec or (f"Page {pg}" if pg is not None else ""))
        for tbl in re.findall(r"\bTable\s+\d+\b", f"{sec} {txt}"):
            tbl_norm = f"Table {tbl.split()[-1]}"
            if tbl_norm not in loc:
                loc = f"{loc} [{tbl_norm}]".strip()
        if loc and loc not in refs:
            refs.append(loc)
    return f"{circular_name}, " + " & ".join(refs) if refs else circular_name


def parse_statutory_threshold_from_chunks(retrieval: RetrievalResponse, fund_type: str) -> StatutoryThresholdSpec:
    """Parse statutory tolerance threshold and EUR indemnification/compensation thresholds from retrieved Parent Chunks."""
    combined = retrieval.combined_text
    primary_chunk = next(
        (chunk for chunk in retrieval.parent_chunks if chunk.doc_id in ("CSSF_02_77", "CSSF_24_856")),
        None,
    )
    if primary_chunk is None:
        raise ValueError("No statutory circular chunks (CSSF_02_77 or CSSF_24_856) found in retrieval response.")
    primary_doc_id, valid_from, valid_to = primary_chunk.doc_id, primary_chunk.valid_from, primary_chunk.valid_to

    ft_low = fund_type.lower()
    if primary_doc_id == "CSSF_02_77":
        if "money market" in ft_low or "mmf" in ft_low or "cash" in ft_low:
            match = _require_match(r"money market UCIs/cash funds:\s*\|\s*(\d+\.\d+)%", combined, "CSSF 02/77 MMF tolerance")
            pct_str, category = match.group(1), "money market UCIs/cash funds"
        elif "bond" in ft_low:
            match = _require_match(r"bond UCIs:\s*\|\s*(\d+\.\d+)%", combined, "CSSF 02/77 Bond tolerance")
            pct_str, category = match.group(1), "bond UCIs"
        elif "mixed" in ft_low:
            match = _require_match(r"mixed UCIs:\s*\|\s*(\d+\.\d+)%", combined, "CSSF 02/77 Mixed tolerance")
            pct_str, category = match.group(1), "mixed UCIs"
        else:
            match = _require_match(r"shares and other financial assets' UCIs.*?\|\s*(\d+\.\d+)%", combined, "CSSF 02/77 Equity tolerance")
            pct_str, category = match.group(1), "shares and other financial assets' UCIs"

        tot_match = re.search(r"indemnification amount does not exceed EUR\s*([\d,]+)", combined, re.I)
        inv_match = re.search(r"reimbursed to an investor does not exceed EUR\s*([\d,]+)", combined, re.I)
        if tot_match and inv_match:
            tot_raw, inv_raw = tot_match.group(1), inv_match.group(1)
        else:
            pair_match = _require_match(
                r"EUR\s*([\d,]+)\s*/\s*EUR\s*([\d,]+)", combined, "CSSF 02/77 EUR indemnification thresholds"
            )
            tot_raw, inv_raw = pair_match.group(1), pair_match.group(2)
        tot_eur = Decimal(tot_raw.replace(",", "") + ".00")
        inv_eur = Decimal(inv_raw.replace(",", "") + ".00")
        pct = Decimal(pct_str)
        dynamic_cit = _build_statutory_citation_from_chunks(
            retrieval.parent_chunks, "CSSF_02_77", "Circular CSSF 02/77"
        )
        section_ids_txt = " ".join(str(getattr(c, "section_id", "") or "") for c in retrieval.parent_chunks)
        roman_sec = re.search(r"Section\s+([IVX]+\.\d+)", section_ids_txt)
        return StatutoryThresholdSpec(
            doc_id="CSSF_02_77",
            circular_name="Circular CSSF 02/77",
            valid_from=valid_from,
            valid_to=valid_to,
            fund_category=category,
            tolerance_threshold_pct=pct,
            tolerance_threshold_bps=round_bps(pct * Decimal("100")),
            indemnification_total_threshold_eur=tot_eur,
            indemnification_single_investor_threshold_eur=inv_eur,
            special_report_rule=(
                f"Under {dynamic_cit}, no full corrective action plan or separate "
                f"special report is required if total indemnification <= EUR {tot_raw} and single investor <= EUR {inv_raw}."
            ),
            restitution_ref=f"Section {roman_sec.group(1)}" if roman_sec else "",
            source_citation=dynamic_cit,
        )

    if "money market" in ft_low or "mmf" in ft_low or "cash" in ft_low:
        match = _require_match(r"Money market fund governed by the MMFR.*?\|\s*(\d+\.\d+)%", combined, "CSSF 24/856 MMF tolerance")
        pct_str = match.group(1)
        category = "Money market fund governed by the MMFR (Table 1)"
    elif "bond" in ft_low:
        match = _require_match(r"bonds and/or other debt.*?\|\s*(\d+\.\d+)%", combined, "CSSF 24/856 Bond tolerance", re.I | re.S)
        pct_str = match.group(1)
        category = "UCITS investing primarily in bonds and/or other debt securities (Table 2)"
    elif "mixed" in ft_low:
        match = _require_match(r"mixed investment policy.*?\|\s*(\d+\.\d+)%", combined, "CSSF 24/856 Mixed tolerance", re.I | re.S)
        pct_str = match.group(1)
        category = "UCITS pursuing a mixed investment policy (Table 2)"
    else:
        match = _require_match(r"shares and/or other.*?\|\s*(\d+\.\d+)%", combined, "CSSF 24/856 Equity tolerance", re.I | re.S)
        pct_str = match.group(1)
        category = "UCITS investing primarily in shares / equity UCIs (Table 2)"

    t9_match = _require_match(
        r"(?:thresholds of EUR\s*([\d,]+)\s*/\s*EUR\s*([\d,]+)|exceeding EUR\s*([\d,]+).*?EUR\s*([\d,]+))",
        combined,
        "CSSF 24/856 Table 9 EUR compensation thresholds",
        re.I | re.S,
    )
    tot_raw = t9_match.group(1) or t9_match.group(3)
    inv_raw = t9_match.group(2) or t9_match.group(4)
    tot_eur = Decimal(tot_raw.replace(",", "") + ".00")
    inv_eur = Decimal(inv_raw.replace(",", "") + ".00")
    pct = Decimal(pct_str)
    dynamic_cit = _build_statutory_citation_from_chunks(
        retrieval.parent_chunks, "CSSF_24_856", "Circular CSSF 24/856"
    )
    tbl_intro = re.search(r"(\d{1,3})\.\s+\*\*(Table \d+)\*\*[^.\n]*significant NAV calculation errors", combined)
    deriv_para = re.search(r"(\d{1,3})\.\s+By way of derogation", combined)
    comp_table_label = tbl_intro.group(2) if tbl_intro else ""
    para_nums = [n for n in (deriv_para.group(1) if deriv_para else "", tbl_intro.group(1) if tbl_intro else "") if n]
    comp_para_refs = f"Para {'/'.join(para_nums)}" if para_nums else ""
    rest_pt = re.search(r"(\d{1,3})\.\s+Where the UCI paid an amount of costs/fees", combined)
    rest_sec = re.search(r"(\d+\.\d+)\.\s+Non-compliant payment of costs/fees", combined)
    restitution_ref = f"Section {rest_sec.group(1)} Point {rest_pt.group(1)}" if (rest_pt and rest_sec) else ""

    sif_entity = bool(re.search(r"\b(?:sif|sicar)s?\b", fund_type.lower()))
    sample_based = False
    if sif_entity:
        row_group = re.search(r"SIF/SICAR.*?exceeding one of the thresholds.*?\|\s*Sample-based review", combined, re.S)
        if row_group is None:
            raise ValueError(
                "Fund type indicates a SIF/SICAR entity under the sample-based-report regime, but the retrieved "
                "summary-table text lacks the SIF/SICAR sample-based row group; refusing to default the attribute. "
                "Widen retrieval or pass an explicit fund type."
            )
        sample_based = True
    return StatutoryThresholdSpec(
        doc_id="CSSF_24_856",
        circular_name="Circular CSSF 24/856",
        valid_from=valid_from,
        valid_to=valid_to,
        fund_category=category,
        tolerance_threshold_pct=pct,
        tolerance_threshold_bps=round_bps(pct * Decimal("100")),
        indemnification_total_threshold_eur=tot_eur,
        indemnification_single_investor_threshold_eur=inv_eur,
        special_report_rule=(
            f"Under {dynamic_cit}, for UCITS/UCI Part II, "
            f"a significant NAV calculation error entailing compensation exceeding EUR {tot_raw} (total) or "
            f"EUR {inv_raw} (single investor), or both thresholds, requires a Special Report (Review of all the errors) by the REA."
        ),
        sample_based_separate_report=sample_based,
        compensation_table_label=comp_table_label,
        compensation_para_refs=comp_para_refs,
        restitution_ref=restitution_ref,
        source_citation=dynamic_cit,
    )


def _parse_tier_footnote(combined: str, marker: str) -> list[FeeTierSpec]:
    """Parse ordered fee bands from the tiered-reduction footnote `marker` references.

    Returns an empty list when the footnote is not a tiered-reduction schedule (e.g. a
    hedging-cost allocation footnote), so non-tier footnotes never fabricate bands.
    """
    fn = re.search(rf"Footnote\s+`?{re.escape(marker)}`?(.*?)(?=\n\*\s+\*\*Footnote|\n#{{1,4}}\s|\Z)", combined, re.S | re.I)
    if fn is None:
        return []
    body = fn.group(1)
    if not (re.search(r"\btiered\b|\breduced rate\b", body, re.I) and "exceeding" in body.lower()):
        return []
    rates = [Decimal(r) for r in re.findall(r"(\d+\.\d+)%\s*per\s+annum", body)]
    caps: list[Decimal] = []
    for raw in re.findall(r"EUR\s*([\d,]+)", body):
        cap = Decimal(raw.replace(",", "") + ".00")
        if cap not in caps:
            caps.append(cap)
    return [
        FeeTierSpec(
            threshold_gav_eur=caps[i] if i < len(caps) else None,
            rate_pct=rate,
            rate_bps=round_bps(rate * Decimal("100")),
        )
        for i, rate in enumerate(rates)
    ]


def _section_anchor_before(combined: str, pos: int) -> str:
    """Return the nearest numbered markdown heading before `pos` as a `Section N.M` anchor (empty when none)."""
    headings = [
        (m.start(), f"Section {m.group(1).rstrip('.')}")
        for m in re.finditer(r"(?:^|\n)#{2,4}\s+(\d+(?:\.\d+)*\.?)\.?\s", combined[:pos])
    ]
    return headings[-1][1] if headings else ""


def parse_prospectus_fee_from_chunks(
    retrieval: RetrievalResponse,
    share_class: str = "Class I",
    prospectus_doc_id: str = "",
    fund_name: str = "",
    registry_title: str = "",
) -> ProspectusFeeSpec:
    """Parse share-class fee schedule (N-band tiered footnotes) and day-count convention for a resolved prospectus doc."""
    if not prospectus_doc_id:
        raise ValueError(
            "No prospectus document ID resolved; pass prospectus_doc_id explicitly or resolve the fund name "
            "against the corpus registry first."
        )
    combined = retrieval.combined_text
    prosp_chunks = [c for c in retrieval.parent_chunks if getattr(c, "doc_id", "") == prospectus_doc_id]
    if not prosp_chunks:
        raise ValueError(f"No {prospectus_doc_id} chunks found in retrieval response.")
    row_match = _require_match(
        rf"{re.escape(share_class)}\b[^\n]*?(?:\*\*)?(\d+\.\d+)%\s*per\s+annum[^\n]*?(?:\*\*)?([^\n]*)",
        combined,
        f"Prospectus base management fee for {share_class}",
    )
    base_pct = Decimal(row_match.group(1))
    row_tail = row_match.group(2).lower()

    tiers: list[FeeTierSpec] = []
    tier_label: str | None = None
    for marker in dict.fromkeys(re.findall(r"\(\*\d+\)", row_tail)):
        if parsed := _parse_tier_footnote(combined, marker):
            tiers, tier_label = parsed, f"Footnote {marker}"
            break
    if not tiers:
        tiers = [FeeTierSpec(threshold_gav_eur=None, rate_pct=base_pct, rate_bps=round_bps(base_pct * Decimal("100")))]

    dc_match = _require_match(r"(Actual|30)/(\d{3})", combined, "Prospectus day-count convention")
    dc_prefix, denom = dc_match.group(1), int(dc_match.group(2))
    dc_convention = f"{dc_prefix}/{denom}"
    fee_sec = _section_anchor_before(combined, row_match.start())
    dc_sec = _section_anchor_before(combined, dc_match.start())
    cit_name = registry_title or fund_name or prospectus_doc_id
    prosp_cit = _build_statutory_citation_from_chunks(retrieval.parent_chunks, prospectus_doc_id, cit_name)
    matched_secs = [s for s in (fee_sec, dc_sec, tier_label or "") if s and s.lower() in combined.lower() and s not in prosp_cit]
    if matched_secs:
        prosp_cit = f"{prosp_cit} ({', '.join(matched_secs)})"
    return ProspectusFeeSpec(
        doc_id=prospectus_doc_id,
        fund_name=fund_name or cit_name,
        share_class=share_class,
        tiers=tiers,
        fee_schedule_section=fee_sec,
        day_count_section=dc_sec,
        tier_footnote_label=tier_label,
        day_count_convention=dc_convention,
        day_count_denominator=denom,
        source_citation=prosp_cit,
    )
