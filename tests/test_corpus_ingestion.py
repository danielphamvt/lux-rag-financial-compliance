"""
Verification & Acceptance Test Suite for Regulatory Corpus Ingestion & Preservation.

Covers Definition of Done (DoD) Criterion 1:
- Verification of all 4 regulatory documents indexed in parent_store (CSSF 02/77, CSSF 24/856,
  UCITS Directive 2009/65/EC, Lux Alpha SICAV Prospectus).
- Atomic preservation of tables, footnotes (*, *1, *2), and soft-hyphen stripping.
- SHA-256 and MD5 cryptographic integrity verification in indexing manifest.
- Anachronistic watermark stripping on CSSF 02/77 Page 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from rag_engine.backends.vector_chunking.ingest import PARENT_STORE_DIR, ComplianceCorpusIngestor


# ---------------------------------------------------------------------------
# Criterion 1: Ingestion, Bitemporal Metadata, & Table/Footnote Preservation
# ---------------------------------------------------------------------------


def test_criterion_1_all_four_documents_indexed(ingestor: ComplianceCorpusIngestor) -> None:
    """Verify that all four regulatory documents are indexed with SHA-256 and MD5 hashes."""
    manifest_path = PARENT_STORE_DIR / "indexing_manifest.json"
    assert manifest_path.exists(), "Indexing manifest must exist in parent_store"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    docs = manifest["documents"]
    assert set(docs.keys()) == {
        "CSSF_02_77",
        "CSSF_24_856",
        "UCITS_2009_65_EC",
        "LUX_ALPHA_PROSPECTUS",
    }
    for doc_id, info in docs.items():
        assert len(info["sha256"]) == 64, f"Missing SHA-256 for {doc_id}"
        assert len(info["md5"]) == 32, f"Missing MD5 for {doc_id}"
        assert info["parent_chunk_count"] > 0
        assert info["child_chunk_count"] > 0


def test_criterion_1_tables_and_footnotes_preserved_in_parent_store(
    ingestor: ComplianceCorpusIngestor,
) -> None:
    """Verify preservation of statutory tolerance tables, footnotes, and formatting across docs."""
    # 1. CSSF 02/77 Page 3 tolerance table
    p_0277_p3 = json.loads((PARENT_STORE_DIR / "CSSF_02_77_p03.json").read_text(encoding="utf-8"))
    txt_p3 = p_0277_p3["page_content"]
    assert "0.25%" in txt_p3 and "0.50%" in txt_p3 and "1.00%" in txt_p3
    assert "Tolerance Threshold (in % of NAV)" in txt_p3

    # Verify Page 1 of CSSF 02/77 has had the anachronistic 2024 repeal watermark stripped
    p_0277_p1 = json.loads((PARENT_STORE_DIR / "CSSF_02_77_p01.json").read_text(encoding="utf-8"))
    assert "24/856" not in p_0277_p1["page_content"]

    # 2. CSSF 24/856 Table 1 (Page 16)
    p_24856_p16 = json.loads((PARENT_STORE_DIR / "CSSF_24_856_p16.json").read_text(encoding="utf-8"))
    assert "Table 1" in p_24856_p16["page_content"]
    assert "0.20%" in p_24856_p16["page_content"]

    # 3. CSSF 24/856 Table 2 & Table 3 + Footnotes (*) (Page 17)
    p_24856_p17 = json.loads((PARENT_STORE_DIR / "CSSF_24_856_p17.json").read_text(encoding="utf-8"))
    txt_p17 = p_24856_p17["page_content"]
    assert "Table 2" in txt_p17 and "Table 3" in txt_p17
    assert "0.50%" in txt_p17 and "1.00%" in txt_p17
    assert "(*)" in txt_p17 and "other assets" in txt_p17.lower()

    # 4. CSSF 24/856 Table 9 + Footnote (*) (Page 42) & Stitched Table 10 + Table 11 (Page 43)
    p_24856_p42 = json.loads((PARENT_STORE_DIR / "CSSF_24_856_p42.json").read_text(encoding="utf-8"))
    txt_p42 = p_24856_p42["page_content"]
    assert "Table 9" in txt_p42 and "EUR 50,000" in txt_p42 and "EUR 5,000" in txt_p42
    assert "(*)" in txt_p42 and "statutory audit" in txt_p42

    p_24856_p43 = json.loads((PARENT_STORE_DIR / "CSSF_24_856_p43.json").read_text(encoding="utf-8"))
    txt_p43 = p_24856_p43["page_content"]
    assert "Table 10" in txt_p43 and "Table 11" in txt_p43
    assert "(*)" in txt_p43 and "EUR 50,000" in txt_p43

    # 5. UCITS Directive Page 23 (Article 22 two-column order & soft-hyphen removal)
    p_ucits_p23 = json.loads((PARENT_STORE_DIR / "UCITS_2009_65_EC_p23.json").read_text(encoding="utf-8"))
    txt_u23 = p_ucits_p23["page_content"]
    assert "\xad" not in txt_u23, "Soft hyphens must be stripped from UCITS Directive"
    assert "Article 22" in txt_u23
    assert "applicable national law and the fund rules" in txt_u23
    assert txt_u23.index("Article 22") < txt_u23.index("(b) ensure that the value of units") < txt_u23.index("Article 23")

    # 6. Lux Alpha Prospectus Section 9.1 Table + Footnotes (*1) & (*2)
    p_prosp_s03 = json.loads((PARENT_STORE_DIR / "LUX_ALPHA_PROSPECTUS_s03.json").read_text(encoding="utf-8"))
    txt_s03 = p_prosp_s03["page_content"]
    assert "0.40%" in txt_s03 and "1.20%" in txt_s03 and "0.30%" in txt_s03
    assert "(*1)" in txt_s03 and "(*2)" in txt_s03 and "EUR 50,000,000" in txt_s03
