"""Corpus PDF & Markdown Page-Tree Parsers (`page_index` backend).

Delegates raw PDF/Markdown text extraction, provenance hashing, and citation
extraction to `rag_engine.core.corpus_parsing` and constructs `page_index` tree nodes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rag_engine.core.corpus_parsing import (
    compute_file_hashes,
    extract_cssf_02_77_pages,
    extract_cssf_24_856_pages,
    extract_generic_pdf_pages,
    extract_outbound_citations as _extract_outbound_citations,
    extract_prospectus_markdown_sections,
    extract_ucits_directive_pages,
)

from .page_index_config import (
    CSSF_02_77_SECTION_TITLES,
    CSSF_24_856_SECTION_TITLES,
    MAX_EUR_AMOUNTS_PER_PAGE,
    MAX_HEADING_CHARS,
    MAX_PERCENTAGES_PER_PAGE,
    MAX_POINTS_PER_PAGE,
    MAX_SUMMARY_HEADINGS,
    MAX_SUMMARY_KEYWORDS,
    OUTBOUND_CITATION_PATTERNS,
    CorpusDocumentSpec,
)


def extract_outbound_citations(text: str) -> list[str]:
    """Extract deduplicated regulatory citations from page Markdown text."""
    return _extract_outbound_citations(text, patterns=OUTBOUND_CITATION_PATTERNS)


def build_page_node(
    spec: CorpusDocumentSpec,
    page_num: int,
    section_id: str,
    md: str,
    hashes: dict[str, str],
    parent_id: str | None = None,
) -> dict[str, Any]:
    """Construct a single page node dictionary for the Document Tree."""
    pid = parent_id or f"{spec.doc_id}_p{page_num:02d}"
    citations = extract_outbound_citations(md)
    headings: list[str] = []
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#") or re.match(
            r"^(?:\*\*)?(?:Table\s+\d+|Article\s+\d+|CHAPTER\s+[IVXLC]+|Section\s+\d+)", s, re.I
        ):
            cleaned_h = re.sub(r"[#*_`]+", "", s).strip()
            if cleaned_h and cleaned_h not in headings:
                headings.append(cleaned_h[:MAX_HEADING_CHARS])

    keywords = list(citations)
    for pt in re.findall(r"(?:^|\n)\s*(\d{1,3})\.\s+", md)[:MAX_POINTS_PER_PAGE]:
        tag = f"Point {pt}"
        if tag not in keywords:
            keywords.append(tag)
    for pct in re.findall(r"\b\d+(?:\.\d+)?\s*%", md)[:MAX_PERCENTAGES_PER_PAGE]:
        if pct not in keywords:
            keywords.append(pct)
    for eur in re.findall(r"EUR\s*[\d,]+", md)[:MAX_EUR_AMOUNTS_PER_PAGE]:
        if eur not in keywords:
            keywords.append(eur)

    summary_parts = [section_id]
    if headings:
        summary_parts.append("Headings: " + "; ".join(headings[:MAX_SUMMARY_HEADINGS]))
    if keywords:
        summary_parts.append("Keywords: " + ", ".join(keywords[:MAX_SUMMARY_KEYWORDS]))

    page_content = (
        f"### [{spec.doc_id}] {section_id} (Page {page_num}, Effective: {spec.valid_from} to {spec.valid_to})\n\n{md}"
    )
    return {
        "parent_id": pid,
        "doc_id": spec.doc_id,
        "source_file": spec.filename,
        "title": spec.title,
        "valid_from": spec.valid_from,
        "valid_to": spec.valid_to,
        "page_number": page_num,
        "section_id": section_id,
        "headings": headings,
        "keywords": keywords,
        "summary": " | ".join(summary_parts),
        "outbound_citations": citations,
        "has_table": "|" in md and "---" in md,
        "has_footnote": "(*)" in md or "(*1)" in md or "(*2)" in md,
        "sha256": hashes["sha256"],
        "md5": hashes["md5"],
        "page_content": page_content,
    }


def _parse_cssf_02_77(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str]) -> list[dict[str, Any]]:
    return [
        build_page_node(spec, page_num, section_id, md, hashes)
        for page_num, section_id, md in extract_cssf_02_77_pages(pdf_path, section_titles=CSSF_02_77_SECTION_TITLES)
    ]


def _parse_cssf_24_856(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str]) -> list[dict[str, Any]]:
    return [
        build_page_node(spec, page_num, section_id, md, hashes)
        for page_num, section_id, md in extract_cssf_24_856_pages(pdf_path, section_titles=CSSF_24_856_SECTION_TITLES)
    ]


def _parse_ucits_directive(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str]) -> list[dict[str, Any]]:
    return [
        build_page_node(spec, page_num, section_id, md, hashes)
        for page_num, section_id, md in extract_ucits_directive_pages(pdf_path)
    ]


def _parse_prospectus_markdown(md_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str]) -> list[dict[str, Any]]:
    raw_text = md_path.read_text(encoding="utf-8")
    doc_header, sections = extract_prospectus_markdown_sections(raw_text)
    pages: list[dict[str, Any]] = []
    for idx, section_id, sub_text_clean in sections:
        body = f"{doc_header}\n\n{sub_text_clean}" if doc_header else sub_text_clean
        pages.append(build_page_node(spec, idx, section_id, body, hashes, parent_id=f"{spec.doc_id}_p{idx:02d}"))
    return pages


def _parse_generic_pdf(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str]) -> list[dict[str, Any]]:
    """Generic PDF-to-page-tree parser for arbitrary multi-document corpora."""
    return [
        build_page_node(spec, page_num, section_id, md, hashes)
        for page_num, section_id, md in extract_generic_pdf_pages(pdf_path, max_heading_chars=MAX_HEADING_CHARS)
    ]
