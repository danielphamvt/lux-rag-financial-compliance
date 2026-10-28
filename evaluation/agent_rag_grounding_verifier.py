"""
RAG Grounding & Telemetry Verification Engine for Compliance Benchmarks.

Provides flexible statutory citation matching (supporting regulatory abbreviations,
basis points vs percent, numeric delimiters, and semantic abstention phrases)
and authentic corpus document retrieval telemetry with multi-document thresholds.
"""

from __future__ import annotations

import re
from typing import Any


def normalize_text_for_matching(text: str) -> str:
    """Normalize text by collapsing whitespace and punctuation delimiters.

    Args:
        text: Input string to normalize.

    Returns:
        Lowercased string with collapsed delimiters and spaces.
    """
    return re.sub(r"[\s\-_/]+", " ", text.lower()).strip()


def match_single_citation(citation: str, text: str) -> bool:
    """Check if a citation or its semantic/regulatory equivalent appears in text.

    Handles:
    - Literal and whitespace-normalized substring matching.
    - Number formatting equivalence (e.g., '50,000' <-> '50000' or '50 000').
    - Basis points vs percent vs 'bps' (e.g., '50 bps' <-> '50 basis points' <-> '0.50%').
    - Statutory section abbreviations (e.g., 'Article 52' <-> 'Art. 52', 'Section 9.1' <-> 'Sec. 9.1').
    - Semantic abstention / out-of-scope equivalents (e.g., 'out of scope' <-> 'outside the scope').

    Args:
        citation: Single citation keyword or phrase.
        text: Response text to audit.

    Returns:
        True if citation or acceptable equivalent appears in text.
    """
    cit_lower = citation.lower().strip()
    text_lower = text.lower()

    # 1. Exact or normalized literal substring
    if cit_lower in text_lower:
        return True

    norm_text = normalize_text_for_matching(text)
    norm_cit = normalize_text_for_matching(citation)
    if norm_cit in norm_text:
        return True

    # 2. Number formatting equivalence: e.g. "50,000" <-> "50000" or "50 000"
    if "," in cit_lower:
        no_comma = cit_lower.replace(",", "")
        if no_comma in text_lower or normalize_text_for_matching(no_comma) in norm_text:
            return True

    # 3. Basis points vs percent vs "bps": e.g. "50 bps" <-> "50 basis points" <-> "0.50%"
    bps_match = re.match(r"^(\d+(?:\.\d+)?)\s*bps$", cit_lower)
    if bps_match:
        val = bps_match.group(1)
        bps_variants = [f"{val} bps", f"{val} basis points", f"{val} bp", f"{val} basis point"]
        try:
            pct_val = float(val) / 100.0
            bps_variants.extend([f"{pct_val:g}%", f"{pct_val:.2f}%"])
        except ValueError:
            pass
        if any(v in text_lower for v in bps_variants):
            return True

    # 4. Regulatory section & circular abbreviations: "Article 52" <-> "Art. 52", "CSSF 24/856" <-> "CSSF Circular 24/856"
    circ_match = re.match(r"^(?:circular\s+)?cssf\s+(?:circular\s+)?(\d{2}/\d{2,4})$", cit_lower)
    if circ_match:
        c_num = circ_match.group(1)
        if any(v in text_lower for v in (f"cssf {c_num}", f"circular cssf {c_num}", f"cssf circular {c_num}")):
            return True

    sec_match = re.match(r"^(article|section|paragraph|point)\s+([\w\.\(\)]+)$", cit_lower)
    if sec_match:
        kind, num = sec_match.group(1), sec_match.group(2)
        abbr_map = {
            "article": ["art.", "art", "article"],
            "section": ["sec.", "sec", "section"],
            "paragraph": ["para.", "para", "paragraph"],
            "point": ["pt.", "pt", "point"],
        }
        for prefix in abbr_map.get(kind, [kind]):
            variant = f"{prefix} {num}"
            if variant in text_lower or normalize_text_for_matching(variant) in norm_text:
                return True

    # 5. Semantic abstention / out-of-scope equivalents
    if cit_lower in ("out of scope", "outside of scope"):
        scope_variants = ("out of scope", "outside of scope", "outside the scope", "not in scope", "beyond scope", "beyond the scope")
        if any(v in text_lower for v in scope_variants):
            return True

    if cit_lower == "not governed":
        gov_variants = ("not governed", "not covered", "not subject to", "not regulated", "not applicable", "does not govern")
        if any(v in text_lower for v in gov_variants):
            return True

    if cit_lower == "same day":
        if any(v in text_lower for v in ("same day", "same-day", "same business day", "intraday")):
            return True

    if cit_lower == "prior to publication":
        if any(v in text_lower for v in ("prior to publication", "before publication", "prior to publishing", "before dissemination")):
            return True

    return False


def verify_citations(expected_citations: list[str], text: str) -> bool:
    """Verify expected statutory citations or keywords appear in text with flexible matching.

    Args:
        expected_citations: List of statutory keywords (e.g. ['CSSF 24/856', 'Article 52']).
        text: Response text to audit.

    Returns:
        True if at least 60% of expected citations match.
    """
    if not expected_citations:
        return True
    matches = sum(1 for c in expected_citations if match_single_citation(c, text))
    return (matches / len(expected_citations)) >= 0.60


def extract_retrieved_documents(search_tool_outputs: list[str]) -> list[dict[str, str]]:
    """Extract retrieved doc_id, file_name, and section from search_documents ToolMessages.

    Args:
        search_tool_outputs: List of raw content strings from search ToolMessages.

    Returns:
        List of dicts with keys 'doc_id', 'file_name', 'section'.
    """
    results = []
    for out in search_tool_outputs or []:
        for block in re.split(r"(?=Parent ID:)", out):
            doc_m = re.search(r"Document ID:\s*([^\n]+)", block)
            file_m = re.search(r"File Name:\s*([^\n]+)", block)
            sec_m = re.search(r"Section / Page:\s*([^\n]+)", block)
            if doc_m or file_m:
                results.append({
                    "doc_id": doc_m.group(1).strip() if doc_m else "",
                    "file_name": file_m.group(1).strip() if file_m else "",
                    "section": sec_m.group(1).strip() if sec_m else "",
                })
    return results


def verify_retrieval_fidelity(
    expected_docs: list[str],
    search_tool_outputs: list[str],
    min_retrieval_ratio: float = 0.60,
) -> tuple[bool, list[str], list[str]]:
    """Verify whether expected statutory documents were actually retrieved by RAG.

    Applies flexible multi-document coverage:
    - 0 expected docs (Abstention): Always passes (True).
    - 1 expected doc: Strict 100% match required.
    - 2 expected docs: Requires at least 1 primary doc (>= 50%).
    - 3+ expected docs: Requires at least min_retrieval_ratio (e.g. 2/3 = 66.7% >= 60%).

    Args:
        expected_docs: List of canonical doc names (e.g. ['CSSF24_856eng.pdf', 'lux_alpha_ucits_prospectus.md']).
        search_tool_outputs: ToolMessage content strings from search_documents.
        min_retrieval_ratio: Minimum ratio of expected documents required for multi-doc queries (default 0.60).

    Returns:
        Tuple of (retrieval_passed: bool, matched_docs: list[str], missing_docs: list[str]).
    """
    if not expected_docs:
        return True, [], []

    retrieved = extract_retrieved_documents(search_tool_outputs)
    retrieved_tokens = set()
    for d in retrieved:
        for fld in (d.get("doc_id", ""), d.get("file_name", "")):
            norm = fld.lower().replace("-", "_").replace(".pdf", "").replace(".md", "")
            retrieved_tokens.add(norm)

    matched = []
    missing = []
    for exp in expected_docs:
        exp_norm = exp.lower().replace("-", "_").replace(".pdf", "").replace(".md", "")
        if any(exp_norm in tok or tok in exp_norm for tok in retrieved_tokens):
            matched.append(exp)
        else:
            missing.append(exp)

    total_expected = len(expected_docs)
    if total_expected <= 1:
        hit = len(matched) == 1
    elif total_expected == 2:
        hit = len(matched) >= 1
    else:
        hit = (len(matched) / total_expected) >= min_retrieval_ratio

    return hit, matched, missing
