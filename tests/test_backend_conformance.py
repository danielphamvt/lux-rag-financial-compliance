"""Backend conformance suite: every registered backend must satisfy the shared contract.

Read-only on production indexes: calls retrieve() and tests isolated temporary directories for
ingestion/cache invalidation, corrupt parent JSON handling, and Neuro-Symbolic NAV Audit edge cases.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import fitz
import pytest

from rag_engine.backends.page_index.ingest import PageIndexIngestor
from rag_engine.backends.page_index.page_index_config import discover_corpus_registry as pi_discover_registry
from rag_engine.backends.page_index.retrieval import PageIndexRetriever
from rag_engine.backends.vector_chunking.ingest import ComplianceCorpusIngestor
from rag_engine.backends.vector_chunking.retrieval import (
    BitemporalComplianceRetriever,
    RetrievedParentChunk,
    RetrievalResponse,
)
from rag_engine.backends.vector_chunking.vector_chunking_config import discover_corpus_registry as vc_discover_registry
from rag_engine.core import nav_audit as nav_audit_mod
from rag_engine.core.corpus_parsing import CORPUS_REGISTRY, CorpusDocumentSpec
from rag_engine.core.nav_audit import FundAccountingPackInput, NeuroSymbolicNAVAuditEngine
from rag_engine.core.registry import get_backend, list_backends
from evaluation.run_benchmark import (
    ABSTENTION_SCORE_THRESHOLDS,
    EVAL_DATASET_PATH,
    evaluate_single_query,
)


@pytest.fixture(scope="session")
def backend_factory():
    built: dict[str, object] = {}

    def factory(name: str):
        if name not in built:
            built[name] = get_backend(name)
        return built[name]

    return factory


@pytest.mark.parametrize("backend_name", list_backends())
class TestBackendConformance:
    def test_retrieval_result_schema(self, backend_factory, backend_name):
        b = backend_factory(backend_name)
        res = b.retrieve("management fee cap Class I shares", audit_date="2025-02-15", top_k=2)
        assert 1 <= len(res.chunks) <= 2  # respects top_k, corpus non-empty
        for c in res.chunks:
            assert c.id and c.text
            assert isinstance(c.score, float)
            assert c.valid_from is None or isinstance(c.valid_from, str)
            assert c.valid_to is None or isinstance(c.valid_to, str)

    def test_respects_audit_date(self, backend_factory, backend_name):
        b = backend_factory(backend_name)
        audit_date = "2024-06-15"
        res = b.retrieve("tolerance threshold NAV management fee", audit_date=audit_date, top_k=5)
        for c in res.chunks:
            if c.valid_from and c.valid_to:
                assert c.valid_from <= audit_date <= c.valid_to

    def test_retrieve_is_deterministic(self, backend_factory, backend_name):
        b = backend_factory(backend_name)
        r1 = b.retrieve("swing pricing dilution factor", audit_date="2025-02-15", top_k=3)
        r2 = b.retrieve("swing pricing dilution factor", audit_date="2025-02-15", top_k=3)
        assert r1 == r2

    def test_invalid_audit_date_raises_value_error(self, backend_factory, backend_name):
        b = backend_factory(backend_name)
        for bad_date in ("2025-2-15", "15/02/2025", "2025-02-30", "not-a-date"):
            with pytest.raises(ValueError, match="Invalid audit_date"):
                b.retrieve("tolerance threshold", audit_date=bad_date)


class TestPageIndexRegression:
    def test_baseline_40_and_out_of_domain_10_abstention(self, backend_factory):
        pi_backend = backend_factory("page_index")
        retriever: PageIndexRetriever = pi_backend.retriever

        all_queries = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))["retrieval_queries"]
        baseline_queries = [
            q for q in all_queries if q.get("challenge_type", "baseline") == "baseline"
        ]
        assert len(baseline_queries) == 40
        for q_item in baseline_queries:
            res = evaluate_single_query(retriever, q_item, mode="hybrid_rerank", top_k=3)
            assert res["strict_pass"], f"Baseline query {q_item['id']} failed: {res['missing_assertions']}"

        ood_queries = [q for q in all_queries if q.get("unanswerable")]
        assert len(ood_queries) == 10
        threshold = ABSTENTION_SCORE_THRESHOLDS["page_index"]
        for q_item in ood_queries:
            res = evaluate_single_query(retriever, q_item, mode="hybrid_rerank", top_k=3)
            assert res["strict_pass"], f"OOD query {q_item['id']} failed to abstain: top_score={res['top_score']}"
            assert res["top_score"] < threshold

        sample = retriever.retrieve("Table 1 MMF tolerance threshold", audit_date="2025-02-15", top_k=1)
        assert sample.parent_chunks
        assert not any(k.startswith("_") for k in sample.parent_chunks[0].metadata)
        raw_tree = json.loads(retriever.ingestor.tree_path.read_text(encoding="utf-8"))
        for doc_node in raw_tree["documents"].values():
            assert "sections" not in doc_node

    def test_invalid_manifest_raises_value_error(self, tmp_path: Path):
        bad_manifest = tmp_path / "corpus_manifest.json"
        bad_manifest.write_text("{invalid json,", encoding="utf-8")
        for discover_fn in (pi_discover_registry, vc_discover_registry):
            with pytest.raises(ValueError, match="Invalid corpus manifest"):
                discover_fn(primary_dir=tmp_path, fallback_dir=tmp_path)

    def test_markdown_fallback_and_cache_staleness_rebuild(self, monkeypatch, tmp_path: Path):
        import rag_engine.backends.page_index.ingest as ingest_mod
        import rag_engine.backends.page_index.page_index_config as cfg_mod

        docs_dir, idx_dir = tmp_path / "docs", tmp_path / "index"
        docs_dir.mkdir()
        monkeypatch.setattr(cfg_mod, "CORPUS_REGISTRY", {})
        monkeypatch.setattr(ingest_mod, "FALLBACK_DATA_DOCS_DIR", docs_dir)
        extra_md = docs_dir / "custom_policy.md"
        extra_md.write_text("# Policy Title\n\nCustom compliance body without h3 headings.", encoding="utf-8")

        ingestor = PageIndexIngestor(data_docs_dir=docs_dir, index_dir=idx_dir)
        manifest_v1 = ingestor.ingest(clean=False)
        assert "CUSTOM_POLICY" in manifest_v1["documents"]
        assert manifest_v1["documents"]["CUSTOM_POLICY"]["page_count"] == 1

        old_sha = manifest_v1["documents"]["CUSTOM_POLICY"]["sha256"]
        extra_md.write_text("## Section A\nBody A\n\n## Section B\nBody B", encoding="utf-8")
        manifest_v2 = ingestor.ingest(clean=False)
        assert manifest_v2["documents"]["CUSTOM_POLICY"]["sha256"] != old_sha
        assert manifest_v2["documents"]["CUSTOM_POLICY"]["page_count"] == 2


class TestVectorChunkingAndNAVAuditRegression:
    def test_generic_pdf_markdown_ingestion_cache_invalidation_and_corrupt_parent(
        self, backend_factory, monkeypatch, tmp_path: Path
    ):
        import rag_engine.backends.vector_chunking.ingest as vc_ingest_mod
        import rag_engine.backends.vector_chunking.vector_chunking_config as vc_cfg_mod

        vc_backend = backend_factory("vector_chunking")
        shared_ingestor = vc_backend.retriever.ingestor
        monkeypatch.setattr(vc_ingest_mod, "HuggingFaceEmbeddings", lambda *a, **kw: shared_ingestor.dense_embeddings)
        monkeypatch.setattr(vc_ingest_mod, "FastEmbedSparse", lambda *a, **kw: shared_ingestor.sparse_embeddings)
        monkeypatch.setattr(vc_cfg_mod, "CORPUS_REGISTRY", {})

        docs_dir, storage_dir = tmp_path / "docs", tmp_path / "storage"
        docs_dir.mkdir()

        pdf_path = docs_dir / "short_circular.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "CHAPTER 1 OVERVIEW\nShort regulatory circular with 1 page (< 47 pages).")
        doc.save(str(pdf_path))
        doc.close()

        md_path = docs_dir / "note_without_h3.md"
        md_path.write_text("# Standalone Note\n\nBody text without level-3 markdown headings.", encoding="utf-8")

        ingestor = ComplianceCorpusIngestor(data_docs_dir=docs_dir, storage_dir=storage_dir, collection_name="test_vc")
        m1 = ingestor.ingest(clean=False)
        assert {"SHORT_CIRCULAR", "NOTE_WITHOUT_H3"} == set(m1["documents"].keys())
        assert (storage_dir / "markdown_docs" / "short_circular.md").exists()
        assert not (storage_dir / "markdown_docs" / "CSSF24_856eng.md").exists()

        old_md_sha = m1["documents"]["NOTE_WITHOUT_H3"]["sha256"]
        md_path.write_text("## Section 1\nUpdated body A\n\n## Section 2\nUpdated body B", encoding="utf-8")
        m2 = ingestor.ingest(clean=False)
        assert m2["documents"]["NOTE_WITHOUT_H3"]["sha256"] != old_md_sha
        assert m2["documents"]["NOTE_WITHOUT_H3"]["parent_chunk_count"] == 2

        corrupt_parent = next(p for p in ingestor.parent_store_dir.glob("*.json") if p.name != "indexing_manifest.json")
        corrupt_parent.write_text("{corrupt json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupt parent chunk JSON"):
            BitemporalComplianceRetriever(ingestor=ingestor)

    def test_nav_audit_edge_cases_and_live_chat_logging(self, backend_factory, tmp_path: Path):
        vc_backend = backend_factory("vector_chunking")
        logs_dir = tmp_path / "logs"
        engine = NeuroSymbolicNAVAuditEngine(retriever=vc_backend.retriever, logs_dir=logs_dir)

        compliant_res = engine.execute_audit(
            FundAccountingPackInput(
                scenario_id="live_chat",
                audit_date="2025-02-15",
                audit_type="fee_accrual_audit",
                fund_name="Lux Alpha SICAV",
                fund_type="Equity UCITS",
                share_class="Class I",
                gav_eur=Decimal("80000000.00"),
                accrual_days=365,
                admin_reported_fee_eur=Decimal("290000.00"),
            )
        )
        assert compliant_res.regulatory_verdict == "COMPLIANT_FEE_ACCRUAL_VERIFIED"
        assert "PROSPECTUS_FEE_SCHEDULE_COMPLIANT" in compliant_res.control_flags
        assert compliant_res.is_material_nav_breach is False
        assert compliant_res.variance_bps == Decimal("0.00")
        assert Path(compliant_res.audit_trail_path).parent == logs_dir / "live_chat"
        assert Path(compliant_res.audit_trail_path).exists()

        under_res = engine.execute_audit(
            FundAccountingPackInput(
                scenario_id="under_accrual",
                audit_date="2025-02-15",
                audit_type="fee_accrual_audit",
                fund_name="Lux Alpha SICAV",
                fund_type="Equity UCITS",
                share_class="Class I",
                gav_eur=Decimal("80000000.00"),
                accrual_days=365,
                admin_reported_fee_eur=Decimal("250000.00"),
            )
        )
        assert under_res.variance_bps == Decimal("5.00")
        assert "understated" in under_res.verdict_rationale

        husd_res = engine.execute_audit(
            FundAccountingPackInput(
                scenario_id="husd_overcharge",
                audit_date="2025-02-15",
                audit_type="fee_accrual_audit",
                fund_name="Lux Alpha SICAV",
                fund_type="Equity UCITS",
                share_class="Class H-USD",
                gav_eur=Decimal("60000000.00"),
                accrual_days=365,
                admin_reported_fee_eur=Decimal("395416.67"),
                admin_used_day_count="Actual/360",
            )
        )
        assert "FOOTNOTE_1_CLASS_H_USD_EXCLUSION_VERIFIED" in husd_res.control_flags

        with pytest.raises(ValueError, match="strictly positive"):
            engine.execute_audit(
                FundAccountingPackInput(
                    scenario_id="zero_shares",
                    audit_date="2025-02-15",
                    audit_type="nav_pricing_error",
                    fund_name="Lux Cash MMF",
                    fund_type="Money Market UCI / MMF",
                    replicated_net_assets_eur=Decimal("100000.00"),
                    shares_outstanding=Decimal("0"),
                    reported_nav_per_share=Decimal("100.00"),
                )
            )


class TestFundAgnosticNAVAudit:
    """Fund-agnostic engine guarantees: N-tier schedules with parameter-true labels, loud resolution failures."""

    MERIDIAN_FEE_MD = """### 5.1. Schedule of Management Fees

| Share Class | Target Investors | Annual Management Fee Rate |
| :--- | :--- | :--- |
| **Class P (Professional)** | Professional Investors | **0.50% per annum (50 bps)** `(*7)` |

* **Footnote `(*7)` — Tiered Fee Reduction for Class P:** For **Class P (Professional)** only, a tiered management fee reduction applies based on the sub-fund's Gross Asset Value (GAV): the rate of **0.50% per annum** applies to the first **EUR 10,000,000** of GAV, a reduced rate of **0.35% per annum** applies to the portion of GAV above **EUR 10,000,000** and up to **EUR 50,000,000**, and a further reduced rate of **0.20% per annum** applies to the portion of GAV exceeding **EUR 50,000,000**.

### 5.2. Day-Count Convention for Fee Accruals

All management fees are accrued using the **`Actual/365` (Fixed)** day-count convention.
"""

    STATUTORY_MD = """1. Tolerance thresholds: UCITS investing primarily in shares and/or other transferable securities | 1.00% of NAV (100 bps)

149. **Table 9** deals with the significant NAV calculation errors under Chapter 4 of this Circular.

**Table 9**

| UCI Category / Error Condition | Separate report | Special report |
| :--- | :--- | :--- |
| Significant NAV calculation error entailing compensation exceeding one of the thresholds of EUR 50,000 / EUR 5,000, or both thresholds | — | Review of all the errors |
"""

    @staticmethod
    def _stub_retriever(responses_by_mode_count: list[RetrievedParentChunk]):
        """Return a retriever stub yielding the same crafted chunks regardless of query."""

        class _Stub:
            name = "vector_chunking"

            def retrieve(self, query, audit_date=None, mode=None, top_k=3, candidate_k=0):
                return RetrievalResponse(
                    query=query,
                    audit_date=audit_date,
                    parent_chunks=list(responses_by_mode_count)[:top_k],
                )

        return _Stub()

    def test_synthetic_three_tier_fund_partitioned_accrual_and_true_labels(self, monkeypatch, tmp_path: Path):
        chunks = [
            RetrievedParentChunk(
                parent_id="meridian_fee", doc_id="MERIDIAN_PROSPECTUS", source_file="meridian_global_prospectus.md",
                valid_from="2023-01-01", valid_to="9999-12-31", page_number=2,
                section_id="5.1. Schedule of Management Fees", page_content=self.MERIDIAN_FEE_MD, rerank_score=1.9,
            ),
            RetrievedParentChunk(
                parent_id="cssf24856_t9", doc_id="CSSF_24_856", source_file="CSSF24_856eng.pdf",
                valid_from="2025-01-01", valid_to="9999-12-31", page_number=42,
                section_id="8.4. Summary Tables on the Intervention of the REA (Table 9)", page_content=self.STATUTORY_MD, rerank_score=1.1,
            ),
        ]
        monkeypatch.setattr(
            nav_audit_mod, "CORPUS_REGISTRY",
            {
                "meridian_global_prospectus.md": CorpusDocumentSpec(
                    "meridian_global_prospectus.md", "MERIDIAN_PROSPECTUS",
                    "Meridian Global Fund — UCITS Prospectus (Extract: Fee Schedule)",
                    "2023-01-01", "9999-12-31", "Luxembourg", "CSSF / Fund Prospectus",
                ),
                "CSSF24_856eng.pdf": CORPUS_REGISTRY["CSSF24_856eng.pdf"],
            },
        )
        engine = NeuroSymbolicNAVAuditEngine(retriever=self._stub_retriever(chunks), logs_dir=tmp_path / "logs")
        res = engine.execute_audit(
            FundAccountingPackInput(
                scenario_id="meridian_3tier",
                audit_date="2025-02-15",
                audit_type="fee_accrual_audit",
                fund_name="Meridian Global Fund",
                fund_type="Equity UCITS",
                share_class="Class P",
                gav_eur=Decimal("60000000.00"),
                accrual_days=365,
                admin_reported_fee_eur=Decimal("300000.00"),
                prospectus_doc_id="MERIDIAN_PROSPECTUS",
            )
        )

        assert res.prospectus_resolved_via == "explicit"
        assert res.variance_bps == Decimal("15.00")
        assert res.regulatory_verdict == "PROSPECTUS_METHODOLOGY_DEVIATION_DETECTED"
        assert res.replicated_value == "EUR 210000.00"
        assert res.absolute_variance == "EUR 90000.00"
        assert "FOOTNOTE_1_TIERED_REDUCTION_OMITTED" in res.control_flags
        step_labels = [s["step"] for s in res.python_decimal_calculation_steps]
        assert step_labels[0] == "1. Tier 1 GAV Fee Calculation (GAV <= EUR 10,000,000 @ 0.50% Actual/365)"
        assert step_labels[1] == "2. Tier 2 GAV Fee Calculation (GAV > EUR 10,000,000 and <= EUR 50,000,000 @ 0.35% Footnote (*7) Actual/365)"
        assert step_labels[2] == "3. Tier 3 GAV Fee Calculation (GAV > EUR 50,000,000 @ 0.20% Footnote (*7) Actual/365)"
        assert any(s.startswith("4. Verification of Fund Administrator Erroneous Formula") and "assumed; administrator convention not provided" in s for s in step_labels)
        assert "Section 5.2" in res.verdict_rationale and "Footnote (*7)" in res.verdict_rationale
        spec = res.extracted_pydantic_parameters["prospectus_fee_spec"]
        assert len(spec["tiers"]) == 3 and spec["fee_schedule_section"] == "Section 5.1" and spec["tier_footnote_label"] == "Footnote (*7)"

    def test_resolution_fails_loudly_for_unknown_and_ambiguous_funds(self, backend_factory, monkeypatch, tmp_path: Path):
        vc_backend = backend_factory("vector_chunking")
        engine = NeuroSymbolicNAVAuditEngine(retriever=vc_backend.retriever, logs_dir=tmp_path / "logs")
        inp = FundAccountingPackInput(
            scenario_id="unknown_fund",
            audit_date="2025-02-15",
            audit_type="fee_accrual_audit",
            fund_name="Meridian Global Fund",
            fund_type="Equity UCITS",
            share_class="Class I",
            gav_eur=Decimal("1000000.00"),
            accrual_days=365,
            admin_reported_fee_eur=Decimal("5000.00"),
        )
        with pytest.raises(ValueError, match="does not uniquely match any prospectus.*Lux Alpha"):
            engine.execute_audit(inp)

        ambiguous_registry = dict(CORPUS_REGISTRY)
        ambiguous_registry["alpha_two.md"] = CorpusDocumentSpec(
            "alpha_two.md", "ALPHA_TWO_PROSPECTUS", "Alpha Two SICAV — UCITS Prospectus (Fee Schedule)",
            "2023-01-01", "9999-12-31", "Luxembourg", "CSSF / Fund Prospectus",
        )
        monkeypatch.setattr(nav_audit_mod, "CORPUS_REGISTRY", ambiguous_registry)
        ambiguous_inp = inp.model_copy(update={"fund_name": "Alpha"})
        with pytest.raises(ValueError, match="matches multiple prospectus documents"):
            engine.execute_audit(ambiguous_inp)
