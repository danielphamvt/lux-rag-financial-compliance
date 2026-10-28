"""Shared corpus discovery, provenance hashing, citation extraction, and PDF/Markdown parsers.

Used by both `vector_chunking` and `page_index` backends so layout normalization,
table reconstruction, footnote preservation, and manifest validation have a single
source of truth.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

DEFAULT_VALID_FROM: str = "1900-01-01"
DEFAULT_VALID_TO: str = "9999-12-31"
DEFAULT_JURISDICTION: str = "Luxembourg / EU"
DEFAULT_REGULATORY_AUTHORITY: str = "Regulatory / Fund Compliance Corpus"
SUPPORTED_CORPUS_EXTENSIONS: tuple[str, ...] = (".pdf", ".md")
CORPUS_MANIFEST_OVERRIDE_FILENAME: str = "corpus_manifest.json"
HASH_READ_BLOCK_SIZE: int = 65536

UCITS_HEADER_MIN_Y0: float = 62.0
UCITS_COLUMN_SPLIT_OFFSET: float = 10.0
UCITS_MAX_ARTICLES_IN_SECTION_ID: int = 4
UCITS_PAGE_23_SECTION_ID: str = (
    "Chapter IV — Obligations Regarding the Depositary: Article 22 (22(1), 22(2), 22(3)(a)-(e)) & Articles 23–26 (Page 23)"
)

OUTBOUND_CITATION_PATTERNS: tuple[str, ...] = (
    r"\bSection\s+\d+(?:\.\d+)*\b", r"\bSection\s+[IVX]+\.\s*\d+\.(?:\s*\([a-z]\))?",
    r"\bChapter\s+\d+(?:\.\d+)*\b", r"\bArticle\s+\d+(?:\(\d+\))?(?:\([a-z]\))?\b",
    r"\bTable\s+\d+\b", r"\bTables\s+\d+,\s*\d+\s+and\s+\d+\b",
    r"\bpoint\s+35\([a-d]\)(?:\((?:i|ii|iii|iv|v)\))?\b", r"\bpoint\s+\([a-d]\)\b",
    r"\bsubpoints\s+\(i\)\s+and\s+\(iii\)\b", r"\bCircular\s+CSSF\s+(?:02/77|24/856|21/790|2000/8)\b",
    r"Footnote\s+`?\(\*\d*\)`?", r"\(\*1\)", r"\(\*2\)", r"\(\*\)",
)

CSSF_02_77_SECTION_TITLES: dict[int, str] = {
    3: "Section I.2 — Materiality & NAV Error Tolerance Thresholds (Page 3)",
    4: "Section I.3(a)-(b) — Correction Procedures & EUR 25,000 / EUR 2,500 Indemnification Thresholds (Page 4)",
    5: "Section I.3(a)-(b) — Correction Procedures & EUR 25,000 / EUR 2,500 Indemnification Thresholds (Page 5)",
    7: "Section I.3(c)-(d) — De Minimis & External Auditor Review EUR 25,000 / EUR 2,500 (Page 7)",
    9: "Section I.3(c)-(d) — De Minimis & External Auditor Review EUR 25,000 / EUR 2,500 (Page 9)",
    10: "Section II — Non-Compliance with Investment Rules (Page 10)",
    11: "Section II — Non-Compliance with Investment Rules (Page 11)",
}

CSSF_24_856_SECTION_TITLES: dict[int, str] = {
    15: "Section 4.1 — Definition of an NAV calculation error (Paragraphs 28–30)",
    16: "Section 4.2(a) — Tolerance threshold for Money Market Funds (MMFs) in Table 1 (Paragraphs 31–35)",
    17: "Section 4.2(b)-(c) — Tolerance thresholds for UCITS (Table 2) and UCIs Part II / ELTIFs (Table 3) with Footnotes (*)",
    18: "Section 4.2 — Point 35(d)(i)-(iii) Customized Tolerance Thresholds & 5% of the NAV Maximum Cap (SIFs/SICARs/UCI Part II)",
    26: "Chapter 5 Section 5.4 — Correction of active non-compliance with investment rules (Paragraphs 65–71, Section 5.5 reference)",
    27: "Chapter 5 Section 5.5 — Correction methods for instances of non-compliance & impact calculation (Paragraphs 72–76)",
    32: "Chapter 6 Section 6.1 — Incorrect application of Swing Pricing (Table 6 & Table 7 with Footnotes (*))",
    33: "Chapter 6 Section 6.1–6.2 — Swing Pricing (Table 8 with Footnote (*)) & Non-compliant payment of costs/fees",
    42: "Chapter 8 Section 8.4 — Summary tables on REA intervention: Table 9 (EUR 50,000 / EUR 5,000 thresholds for separate/special report)",
    43: "Chapter 8 Section 8.4 — Summary tables on REA intervention: Table 10 & Table 11 (EUR 50,000 / EUR 5,000 thresholds)",
    46: "Chapter 10 — Transitional & Entry into Force Provisions (Paragraphs 162–165: 29 March 2024 to 1 January 2025)",
    47: "Chapter 10 — Repeal of Circular CSSF 02/77 (Paragraph 166)",
}


@dataclass(frozen=True)
class CorpusDocumentSpec:
    """Immutable specification and bitemporal validity window for an ingested corpus document."""

    filename: str
    doc_id: str
    title: str
    valid_from: str = DEFAULT_VALID_FROM
    valid_to: str = DEFAULT_VALID_TO
    jurisdiction: str = DEFAULT_JURISDICTION
    regulatory_authority: str = DEFAULT_REGULATORY_AUTHORITY

    @property
    def valid_from_int(self) -> int:
        return int(self.valid_from.replace("-", ""))

    @property
    def valid_to_int(self) -> int:
        return int(self.valid_to.replace("-", ""))


CORPUS_REGISTRY: dict[str, CorpusDocumentSpec] = {
    "cssf02_77eng.pdf": CorpusDocumentSpec(
        "cssf02_77eng.pdf", "CSSF_02_77",
        "Circular CSSF 02/77 — Protection of investors in case of NAV calculation error and non-compliance with investment rules",
        "2002-11-27", "2024-12-31", "Luxembourg", "CSSF",
    ),
    "CSSF24_856eng.pdf": CorpusDocumentSpec(
        "CSSF24_856eng.pdf", "CSSF_24_856",
        "Circular CSSF 24/856 — Protection of investors in case of an NAV calculation error, an instance of non-compliance with the investment rules and other errors at UCI level",
        "2025-01-01", "9999-12-31", "Luxembourg", "CSSF",
    ),
    "UCITS_Directive_2009_65_EC.pdf": CorpusDocumentSpec(
        "UCITS_Directive_2009_65_EC.pdf", "UCITS_2009_65_EC",
        "Directive 2009/65/EC of the European Parliament and of the Council (UCITS IV Directive)",
        "2009-11-17", "9999-12-31", "European Union", "European Parliament / Council of the EU",
    ),
    "lux_alpha_ucits_prospectus.md": CorpusDocumentSpec(
        "lux_alpha_ucits_prospectus.md", "LUX_ALPHA_PROSPECTUS",
        "Lux Alpha SICAV — UCITS Prospectus (Extract: Valuation & Fee Schedule)",
        "2023-01-01", "9999-12-31", "Luxembourg", "CSSF / Fund Prospectus",
    ),
}


def compute_file_hashes(file_path: Path) -> dict[str, str]:
    """Compute SHA-256 and MD5 hex digests for immutable audit provenance."""
    sha256_hash, md5_hash = hashlib.sha256(), hashlib.md5()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(HASH_READ_BLOCK_SIZE), b""):
            sha256_hash.update(block)
            md5_hash.update(block)
    return {"sha256": sha256_hash.hexdigest(), "md5": md5_hash.hexdigest()}


def extract_outbound_citations(text: str, patterns: tuple[str, ...] = OUTBOUND_CITATION_PATTERNS) -> list[str]:
    """Extract deduplicated regulatory citations from Markdown text."""
    found: list[str] = []
    for pat in patterns:
        for match in re.findall(pat, text, flags=re.IGNORECASE):
            cleaned = re.sub(r"\s+", " ", match.strip().strip("`"))
            if cleaned not in found:
                found.append(cleaned)
    return found


def discover_corpus_registry(
    primary_dir: Path, fallback_dir: Path, default_registry: dict[str, CorpusDocumentSpec] | None = None
) -> dict[str, CorpusDocumentSpec]:
    """Build the document registry from `default_registry`, `corpus_manifest.json`, and directory scan."""
    base_reg = CORPUS_REGISTRY if default_registry is None else default_registry
    existing = [f for f in (primary_dir, fallback_dir) if f.exists()]
    registry: dict[str, CorpusDocumentSpec] = (
        {fn: s for fn, s in base_reg.items() if any((d / fn).is_file() for d in existing)} if existing else dict(base_reg)
    )
    for folder in existing:
        manifest_file = folder / CORPUS_MANIFEST_OVERRIDE_FILENAME
        if manifest_file.is_file():
            try:
                raw_overrides = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"Invalid corpus manifest in {manifest_file}: {exc}") from exc
            if not isinstance(raw_overrides, dict):
                raise ValueError(f"Invalid corpus manifest in {manifest_file}: expected JSON object")
            for fname, meta in raw_overrides.items():
                if not isinstance(meta, dict):
                    raise ValueError(f"Invalid corpus manifest entry for '{fname}' in {manifest_file}")
                valid_from = str(meta.get("valid_from") or DEFAULT_VALID_FROM)
                valid_to = str(meta.get("valid_to") or DEFAULT_VALID_TO)
                try:
                    d_from, d_to = datetime.date.fromisoformat(valid_from), datetime.date.fromisoformat(valid_to)
                except ValueError as exc:
                    raise ValueError(f"Invalid corpus manifest date for '{fname}' in {manifest_file}: {exc}") from exc
                if d_from > d_to:
                    raise ValueError(f"Invalid corpus manifest date range for '{fname}' in {manifest_file}: {valid_from} > {valid_to}")
                registry[fname] = CorpusDocumentSpec(
                    filename=fname,
                    doc_id=str(meta.get("doc_id") or re.sub(r"[^A-Za-z0-9]+", "_", Path(fname).stem).upper()),
                    title=str(meta.get("title") or Path(fname).stem),
                    valid_from=valid_from,
                    valid_to=valid_to,
                    jurisdiction=str(meta.get("jurisdiction") or DEFAULT_JURISDICTION),
                    regulatory_authority=str(meta.get("regulatory_authority") or DEFAULT_REGULATORY_AUTHORITY),
                )
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_CORPUS_EXTENSIONS and path.name not in registry:
                derived_id = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_").upper() or "DOCUMENT"
                registry[path.name] = CorpusDocumentSpec(
                    filename=path.name, doc_id=derived_id, title=path.stem.replace("_", " ").replace("-", " ")
                )
    return registry


def _page_to_clean_md(doc: pymupdf.Document, page_idx: int) -> str:
    raw = pymupdf4llm.to_markdown(doc, pages=[page_idx], header=False, footer=False, ignore_images=True, write_images=False)
    return raw.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


def extract_cssf_02_77_pages(
    pdf_path: Path, section_titles: dict[int, str] = CSSF_02_77_SECTION_TITLES
) -> list[tuple[int, str, str]]:
    """Extract `(page_num, section_id, md)` tuples for `cssf02_77eng.pdf`."""
    pages: list[tuple[int, str, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            md = re.sub(r"^\s*Repealed by Circular CSSF 24/856\s*", "", _page_to_clean_md(doc, page_idx), flags=re.I | re.M)
            if page_num == 3 and "|money market UCIs/cash funds:|0.25% of NAV|" in md:
                old_table = (
                    "|money market UCIs/cash funds:|0.25% of NAV|\n|---|---|\n"
                    "|bond UCIs:|0.50% of NAV|\n|shares and other financial assets' UCIs:|1.00% of NAV|\n|mixed UCIs:|0.50% of NAV.|"
                )
                new_table = (
                    "| UCI Type / Investment Policy | Tolerance Threshold (in % of NAV) |\n| :--- | :--- |\n"
                    "| money market UCIs/cash funds: | 0.25% of NAV (25 bps) |\n| bond UCIs: | 0.50% of NAV (50 bps) |\n"
                    "| shares and other financial assets' UCIs (equity funds): | 1.00% of NAV (100 bps) |\n| mixed UCIs: | 0.50% of NAV. (50 bps) |"
                )
                md = md.replace(old_table, new_table)
            md = re.sub(r"^#\s+\*\*(The tolerance levels which are provided for NAV calculation errors.+?)\*\*", r"**\1**", md, flags=re.M).strip()
            pages.append((page_num, section_titles.get(page_num, f"Page {page_num}"), md))
    return pages


def extract_cssf_24_856_pages(
    pdf_path: Path, section_titles: dict[int, str] = CSSF_24_856_SECTION_TITLES
) -> list[tuple[int, str, str]]:
    """Extract `(page_num, section_id, md)` tuples for `CSSF24_856eng.pdf` with stitched tables."""
    raw_pages: dict[int, str] = {}
    with pymupdf.open(pdf_path) as doc:
        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            md = re.sub(r"CIRCULAR CSSF 24/856\s+\d+/47", "", _page_to_clean_md(doc, page_idx))
            md = re.sub(r"^(?:#+\s*|\s*-\s*)Table\s+(\d+)\s*$", r"**Table \1**", md, flags=re.M)
            md = re.sub(
                r"#\s*\*\*_\(c\)\s*UCIs Part II \(except for MMFs\) and ELTIFs whose units can be held by_\*\*\s*\n+#\s*\*\*_investors that are not well-informed or professional investors_\*\*",
                "## **(c) UCIs Part II (except for MMFs) and ELTIFs whose units can be held by investors that are not well-informed or professional investors**",
                md,
            )
            if page_num == 16 and "Table 1" in md:
                md = re.sub(
                    r"\|\*\*Principal investment policy as defined in the\*\*\|\*\*Tolerance threshold\*\*\|\n\|---\|---\|\n\|\*\*prospectus\*\*\|\*\*\(in % of NAV\)\*\*\|\n\|Money market fund governed by the MMFR\|0\.20%\|",
                    "| **Principal investment policy as defined in the prospectus** | **Tolerance threshold (in % of NAV)** |\n| :--- | :--- |\n| Money market fund governed by the MMFR (MMF) | 0.20% (20 bps) |",
                    md,
                )
            raw_pages[page_num] = md.strip()
    if 42 in raw_pages:
        raw_pages[42] = (
            "### Section 8.4 — Summary Tables on the Intervention of the REA (Table 9 & Table 10)\n\n"
            "149. **Table 9** deals with the significant NAV calculation errors under Chapter 4 of this Circular:\n\n**Table 9**\n\n"
            "| UCI Category / Error Condition | **Separate report** | **Special report** |\n| :--- | :--- | :--- |\n"
            "| **_UCITS/UCI Part II_** | | |\n"
            "| Significant NAV calculation error entailing compensation exceeding one of the thresholds of EUR 50,000 / EUR 5,000, or both thresholds | — | Review of all the errors |\n"
            "| Significant NAV calculation error entailing compensation that did neither exceed the EUR 50,000 threshold, nor the EUR 5,000 threshold | Sample-based review of the errors | — |\n"
            "| Significant NAV calculation error in a closed-ended UCI (UCI Part II) | No intervention (*) | — |\n"
            "| **_SIF/SICAR_** | | |\n"
            "| Significant NAV calculation error entailing compensation exceeding one of the thresholds of EUR 50,000 / EUR 5,000 or both thresholds | Sample-based review of the errors | — |\n"
            "| Significant NAV calculation error entailing compensation that did neither exceed the EUR 50,000 threshold, nor the EUR 5,000 threshold | Sample-based review of the errors | — |\n"
            "| Significant NAV calculation error in a closed-ended UCI | No intervention (*) | — |\n\n"
            "(*) The REA verifies, as part of its statutory audit, that the NAV accurately reflects the value of the UCI’s assets and liabilities calculated in line with the constitutive documents and/or the prospectus of the UCI."
        )
    if 43 in raw_pages:
        raw_pages[43] = (
            "### Section 8.4 — Summary Tables on the Intervention of the REA (Table 10 & Table 11)\n\n"
            "150. **Table 10** deals with the instances of non-compliance with the investment rules under Chapter 5 of this Circular:\n\n**Table 10** (Stitched across Pages 42–43)\n\n"
            "| UCI Category / Non-Compliance Condition | **Separate report** | **Special report** |\n| :--- | :--- | :--- |\n"
            "| **_UCITS/UCI Part II_** | | |\n"
            "| Active non-compliance with the investment rules entailing compensation exceeding one of the thresholds of EUR 50,000/EUR 5,000 (*), or both thresholds | — | Review of all the instances of non-compliance |\n"
            "| Active non-compliance with the investment rules entailing compensation that did neither exceed the EUR 50,000 threshold, nor the EUR 5,000 threshold (*) | Sample-based review of the instances of non-compliance | — |\n"
            "| Passive non-compliance with the investment rules | Sample-based review of the instances of non-compliance | — |\n"
            "| **_SIF/SICAR_** | | |\n"
            "| Active non-compliance with the investment rules | Sample-based review of the instances of non-compliance | — |\n"
            "| Passive non-compliance with the investment rules | Sample-based review of the instances of non-compliance | — |\n\n"
            "(*) The EUR 5,000 threshold for compensation of one single investor applies where the instance of non-compliance with the investment rule exceeds the tolerance threshold applicable to the NAV errors entailing a significant calculation error within the meaning of Chapter 4 of this Circular.\n\n"
            "151. **Table 11** deals with the other errors under Chapter 6 of this Circular:\n\n**Table 11**\n\n"
            "| UCI Category / Other Error Condition | **Separate report** | **Special report** |\n| :--- | :--- | :--- |\n"
            "| **_UCITS/UCI Part II_** | | |\n"
            "| Other error entailing compensation exceeding one of the thresholds of EUR 50,000/EUR 5,000, or both thresholds | — | Review of all the other errors |\n"
            "| Other error entailing compensation that did neither exceed the EUR 50,000 threshold, nor the EUR 5,000 threshold | Sample-based review of the other errors | — |\n"
            "| **_SIF/SICAR_** | | |\n"
            "| Other error entailing compensation exceeding one of the thresholds of EUR 50,000/EUR 5,000, or both thresholds | Sample-based review of the other errors | — |\n"
            "| Other error entailing compensation that did neither exceed the EUR 50,000 threshold, nor the EUR 5,000 threshold | Sample-based review of the other errors | — |"
        )
    if 31 in raw_pages and 32 in raw_pages:
        raw_pages[32] = "99. **Table 6** hereafter specifies the case of non-application of the swing factor as set for the UCI:\n\n" + raw_pages[32]
    if 46 in raw_pages and 47 in raw_pages:
        raw_pages[46] = (
            raw_pages[46]
            + "\n\nNote on publication date: Circular CSSF 24/856 was published on 29 March 2024 (signed 28 March 2024) and enters into force on 1 January 2025. For NAV errors detected between 29 March 2024 and 1 January 2025, Circular CSSF 02/77 continues to apply.\n\n"
            + raw_pages[47]
        )
    return [(p, section_titles.get(p, f"Page {p}"), raw_pages[p]) for p in sorted(raw_pages.keys())]


def extract_ucits_directive_pages(pdf_path: Path) -> list[tuple[int, str, str]]:
    """Extract `(page_num, section_id, md)` tuples for `UCITS_Directive_2009_65_EC.pdf`."""
    pages: list[tuple[int, str, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_idx in range(len(doc)):
            page_num, page = page_idx + 1, doc[page_idx]
            mid_x = page.rect.width / 2.0
            blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[1] >= UCITS_HEADER_MIN_Y0]
            left = sorted([b for b in blocks if b[0] < mid_x - UCITS_COLUMN_SPLIT_OFFSET], key=lambda b: (round(b[1], 1), round(b[0], 1)))
            right = sorted([b for b in blocks if b[0] >= mid_x - UCITS_COLUMN_SPLIT_OFFSET], key=lambda b: (round(b[1], 1), round(b[0], 1)))
            ordered_lines: list[str] = []
            for b in left + right:
                blk_text = re.sub(r"\s*\n\s*", " ", re.sub(r"\xad\s*", "", b[4].replace("\xa0", " "))).strip()
                if not blk_text:
                    continue
                if re.match(r"^CHAPTER\s+[IVXLC]+$", blk_text, flags=re.I):
                    ordered_lines.append(f"## {blk_text}")
                elif re.match(r"^SECTION\s+\d+$", blk_text, flags=re.I) or re.match(r"^Article\s+\d+[a-z]?$", blk_text):
                    ordered_lines.append(f"### {blk_text}")
                else:
                    ordered_lines.append(blk_text)
            md = "\n\n".join(ordered_lines).strip()
            articles = list(dict.fromkeys(re.findall(r"\bArticle\s+\d+[a-z]?\b", md)))
            section_id = f"Page {page_num} ({', '.join(articles[:UCITS_MAX_ARTICLES_IN_SECTION_ID])})" if articles else f"Page {page_num}"
            pages.append((page_num, UCITS_PAGE_23_SECTION_ID if page_num == 23 else section_id, md))
    return pages


def extract_prospectus_markdown_sections(raw_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Extract `(doc_header, [(idx, section_id, sub_text_clean), ...])` from Markdown text."""
    parts = re.split(r"(?=^###\s+)", raw_text, flags=re.M)
    preamble, subsections = parts[0].strip(), parts[1:]
    if not subsections and raw_text.strip():
        parts_h2 = re.split(r"(?=^##\s+)", raw_text, flags=re.M)
        preamble, subsections = (parts_h2[0].strip(), parts_h2[1:]) if len(parts_h2) > 1 else ("", [raw_text.strip()])
    doc_header = "\n".join(line for line in preamble.splitlines() if line.startswith("# ") or line.startswith("**"))
    sections = [
        (idx, sub.strip().splitlines()[0].lstrip("#").strip() or f"Page {idx}", sub.strip())
        for idx, sub in enumerate(subsections, start=1) if sub.strip()
    ]
    return doc_header, sections


def extract_generic_pdf_pages(pdf_path: Path, max_heading_chars: int = 120) -> list[tuple[int, str, str]]:
    """Extract `(page_num, section_id, md)` tuples for arbitrary non-baseline PDFs."""
    pages: list[tuple[int, str, str]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_idx in range(len(doc)):
            page_num, md = page_idx + 1, _page_to_clean_md(doc, page_idx).strip()
            first_h = next((ln.lstrip("#").strip() for ln in md.splitlines() if ln.strip().startswith("#")), f"Page {page_num}")
            pages.append((page_num, (first_h or f"Page {page_num}")[:max_heading_chars], md))
    return pages
