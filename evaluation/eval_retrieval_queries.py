"""
Verification & Acceptance Test Suite for 100 Retrieval Queries (Benchmark v4.0.0).

Covers Definition of Done (DoD) Criteria:
- Criterion 2: 40 Baseline Evaluation Queries (Q01-Q40 across 6 Taxonomy Groups) 100% Accuracy & Zero Temporal Leakage.
- Criterion 2b: 60 Stress Evaluation Queries (Q41-Q100) Zero Temporal Leakage & Table-Footnote Fusion.
- Criterion 6: Calibrated Abstention on Adversarial Out-of-Domain Queries (Q81-Q90).
- Criterion 7: Anti-Spurious Entity-Context Proximity & Schema Resilience.
- Featured 10 UI Demo Queries: Strict 100% pass verification against frontend profile.
- Criterion 8: 3-Embedding Model Comparison (Nomic v1.5 vs Qwen3-0.6B vs OpenAI text-embedding-3-small).

Usage Guide:
============
1. Run with pytest (Recommended):
       pytest evaluation/eval_retrieval_queries.py -v
   or:
       python -m pytest evaluation/eval_retrieval_queries.py -v

2. Run directly via Python module / CLI script:
       python -m evaluation.eval_retrieval_queries
   or:
       python evaluation/eval_retrieval_queries.py

3. Filter by Specific Criteria / Query ID:
   - Run 40 baseline queries (Criterion 2):
       pytest evaluation/eval_retrieval_queries.py -k "test_criterion_2_" -v
   - Run 60 stress queries (Criterion 2b):
       pytest evaluation/eval_retrieval_queries.py -k "test_criterion_2b" -v
   - Run 10 out-of-domain abstention queries (Criterion 6):
       pytest evaluation/eval_retrieval_queries.py -k "test_criterion_6" -v
   - Run a single query by ID (e.g., Q01 or Q81):
       pytest evaluation/eval_retrieval_queries.py -k "Q01" -v
   - Run 3-embedding model comparison (Criterion 8):
       pytest evaluation/eval_retrieval_queries.py -k "test_criterion_8" -v

4. Run via Makefile:
       make eval ARGS="evaluation/eval_retrieval_queries.py -v"
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

from evaluation.eval_common import WORKSPACE_ROOT, ensure_runtime

# Strictly enforce isolated test environment (.env.test)
ensure_runtime()

import pytest

import rag_engine.backends.vector_chunking.ingest as ing
from rag_engine.backends.vector_chunking.retrieval import BitemporalComplianceRetriever
from evaluation.conftest import load_full_queries
from evaluation.run_benchmark import (
    ABSTENTION_SCORE_THRESHOLDS,
    _verify_entity_context_assertion,
    normalize_query_item_v4,
    run_ablation_study,
)

_ALL_EVAL_QUERIES = load_full_queries()
_BASELINE_EVAL_QUERIES = [
    q for q in _ALL_EVAL_QUERIES if q.get("challenge_type", "baseline") == "baseline"
]
_STRESS_EVAL_QUERIES = [
    q for q in _ALL_EVAL_QUERIES if q.get("challenge_type", "baseline") != "baseline"
]
_UNANSWERABLE_EVAL_QUERIES = [q for q in _ALL_EVAL_QUERIES if q.get("unanswerable")]


# ---------------------------------------------------------------------------
# Criterion 2: Dynamic Evaluation Queries Pass Rate & Zero Temporal Leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_item", _BASELINE_EVAL_QUERIES, ids=lambda q: q["id"])
def test_criterion_2_evaluation_queries_pass_100_percent(
    ablation_results: dict[str, Any],
    q_item: dict[str, Any],
) -> None:
    """Verify that all 40 baseline evaluation queries pass with zero temporal leakage."""
    per_query = {r["id"]: r for r in ablation_results["hybrid_rerank"]["queries"]}
    res = per_query[q_item["id"]]

    assert res["source_hit"], (
        f"{q_item['id']} missed expected source {q_item['expected_source']}: got {res['retrieved_doc_ids']}"
    )
    assert not res["has_forbidden_source"], (
        f"{q_item['id']} suffered Temporal Leakage! Forbidden source {q_item['forbidden_source']} "
        f"found in {res['retrieved_doc_ids']}"
    )
    assert res["all_assertions_matched"], (
        f"{q_item['id']} missed ground-truth assertions {res['missing_assertions']}"
    )
    assert res["eca_coverage"] == 1.0, f"{q_item['id']} failed entity-context proximity assertion"
    assert res["strict_pass"] is True


@pytest.mark.parametrize("q_item", _STRESS_EVAL_QUERIES, ids=lambda q: q["id"])
def test_criterion_2b_stress_queries_zero_temporal_leakage_and_table_fusion(
    ablation_results: dict[str, Any],
    q_item: dict[str, Any],
) -> None:
    """Verify zero temporal leakage and table-footnote fusion on all 60 stress queries."""
    per_query = {r["id"]: r for r in ablation_results["hybrid_rerank"]["queries"]}
    res = per_query[q_item["id"]]

    assert not res["has_forbidden_source"], (
        f"Stress query {q_item['id']} suffered Temporal Leakage! Forbidden source {q_item['forbidden_source']} "
        f"found in {res['retrieved_doc_ids']}"
    )
    if q_item.get("challenge_type") == "table_footnote_fusion":
        assert res["strict_pass"] is True, (
            f"Table-Footnote Fusion query {q_item['id']} failed: {res['missing_assertions']}"
        )


# ---------------------------------------------------------------------------
# Criterion 6: Adversarial Out-of-Domain Calibrated Abstention Gate (Q81-Q90)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_item", _UNANSWERABLE_EVAL_QUERIES, ids=lambda q: q["id"])
def test_criterion_6_adversarial_out_of_domain_abstention(
    ablation_results: dict[str, Any],
    q_item: dict[str, Any],
) -> None:
    """Verify 100% calibrated abstention across all out-of-domain adversarial queries (Q81-Q90)."""
    per_query = {r["id"]: r for r in ablation_results["hybrid_rerank"]["queries"]}
    res = per_query[q_item["id"]]
    threshold = ABSTENTION_SCORE_THRESHOLDS["hybrid_rerank"]

    assert res["top_score"] < threshold, (
        f"Adversarial query {q_item['id']} exceeded calibrated abstention threshold: "
        f"{res['top_score']} >= {threshold}"
    )
    assert res["strict_pass"] is True, f"Adversarial query {q_item['id']} failed abstention check: {res}"


# ---------------------------------------------------------------------------
# Criterion 7: Anti-Spurious Entity-Context Proximity & Multi-Hop Step Attribution
# ---------------------------------------------------------------------------


def test_criterion_7_entity_context_proximity_and_schema_resilience() -> None:
    """Verify anti-spurious proximity matching and backward-compatible v4 schema normalization."""
    eca_bond = {
        "assertion_id": "ECA_Q04_01",
        "anchor_entity": "bond ucis",
        "target_value": "0.50%",
        "max_char_window": 120,
    }
    spurious_chunk = (
        "table 7 swing factor of 0.50% applied on redemptions. "
        + ("padding text " * 30)
        + "unrelated reference to bond ucis in a distant paragraph."
    )
    valid_chunk = "| bond ucis: | 0.50% of nav (50 bps) |"

    assert _verify_entity_context_assertion(spurious_chunk, eca_bond) is False
    assert _verify_entity_context_assertion(valid_chunk, eca_bond) is True

    # Schema resilience check: missing optional fields never raise KeyError
    minimal_q = {"id": "Q_MIN", "query": "test query"}
    norm_q = normalize_query_item_v4(minimal_q)
    assert norm_q["id"] == "Q_MIN"
    assert norm_q["must_contain"] == []
    assert norm_q["entity_context_assertions"] == []
    assert norm_q["step_evidence_map"] == []


def test_featured_10_ui_demo_queries_pass_100_percent(
    ablation_results: dict[str, Any],
    eval_dataset: dict[str, Any],
) -> None:
    """Verify that the 10 curated demo queries in the UI profile match the benchmark and pass 100%."""
    profile_path = WORKSPACE_ROOT / "frontend" / "profiles" / "financial_compliance" / "profile.json"
    profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
    demo_questions = profile_data.get("sample_questions", [])
    assert len(demo_questions) == 10, f"Expected 10 demo questions, got {len(demo_questions)}"

    def _sig_tokens(text: str) -> set[str]:
        t_low = text.lower()
        return set(re.findall(r"\bclass\s+[a-z0-9-]+\b|[a-z0-9]{4,}", t_low))

    all_qs = eval_dataset["retrieval_queries"]
    q_tokens_map = {q["id"]: (q, _sig_tokens(q["query"])) for q in all_qs}

    def _match_demo_question(demo_q: str) -> dict[str, Any] | None:
        d_low = demo_q.lower()
        for q in all_qs:
            q_low = q["query"].lower()
            if q_low in d_low or d_low in q_low:
                return q
        d_toks = _sig_tokens(demo_q)
        d_classes = {t for t in d_toks if t.startswith("class ")}
        best_q, best_score = None, 0.0
        for _, (q, q_toks) in q_tokens_map.items():
            if not q_toks or not d_toks:
                continue
            q_classes = {t for t in q_toks if t.startswith("class ")}
            if d_classes and q_classes != d_classes:
                continue
            score = len(d_toks & q_toks) / len(q_toks)
            if score > best_score:
                best_q, best_score = q, score
        return best_q if best_score >= 0.45 else None

    expected_demo_ids = {"Q01", "Q02", "Q03", "Q06", "Q22", "Q07", "Q21", "Q34", "Q40", "Q09"}
    per_query = {r["id"]: r for r in ablation_results["hybrid_rerank"]["queries"]}
    matched_ids = []

    for q_text in demo_questions:
        q_item = _match_demo_question(q_text)
        assert q_item is not None, f"Demo question not found in eval_dataset: {q_text}"
        matched_ids.append(q_item["id"])
        res = per_query[q_item["id"]]
        assert res["strict_pass"] is True, (
            f"Featured Demo Query {q_item['id']} failed strict pass! "
            f"Missing: {res['missing_assertions']}, Forbidden: {res['has_forbidden_source']}"
        )

    assert set(matched_ids) == expected_demo_ids and len(matched_ids) == 10


# ---------------------------------------------------------------------------
# Criterion 8: 3-Embedding Model Comparison (Nomic v1.5 vs Qwen3-0.6B vs OpenAI text-embedding-3-small)
# ---------------------------------------------------------------------------


def test_criterion_8_three_embedding_models_comparison(
    ablation_results: dict[str, Any],
    eval_dataset: dict[str, Any],
) -> None:
    """Verify ComplianceCorpusIngestor initialization branches and compare 3 embedding models."""
    # 1. Unit verification of the 3 initialization branches in ComplianceCorpusIngestor
    with patch.object(ing, "FastEmbedSparse"), patch.object(ing, "open_qdrant_with_snapshot"):
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "AZURE_OPENAI_API_KEY": "test-azure-key",
                "EMBEDDING_DIMENSIONS": "768",
            },
            clear=False,
        ), patch.object(ing, "EMBEDDING_MODEL_NAME", "text-embedding-3-small"):
            az_ing = ing.ComplianceCorpusIngestor(storage_dir=ing.STORAGE_DIR / "openai_eval")
            assert az_ing.dense_embeddings.__class__.__name__ == "AzureOpenAIEmbeddings"
            assert az_ing.dense_embeddings.dimensions == 768

        env_openai = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_")}
        env_openai["OPENAI_API_KEY"] = "sk-test-key"
        env_openai.pop("EMBEDDING_DIMENSIONS", None)
        with patch.dict(os.environ, env_openai, clear=True), patch.object(
            ing, "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ):
            oa_ing = ing.ComplianceCorpusIngestor(storage_dir=ing.STORAGE_DIR / "openai_eval")
            assert oa_ing.dense_embeddings.__class__.__name__ == "OpenAIEmbeddings"
            assert oa_ing.dense_embeddings.dimensions is None

    # 2. Load or compute 100-query benchmark ablation for OpenAI `text-embedding-3-small`
    openai_eval_dir = ing.STORAGE_DIR / "openai_eval"
    openai_cache_path = openai_eval_dir / "openai_ablation_results.json"
    openai_ablation = None

    if openai_cache_path.exists():
        cached_oa = json.loads(openai_cache_path.read_text(encoding="utf-8"))
        if cached_oa.get("benchmark_version") == eval_dataset.get("metadata", {}).get("version", "4.0.0"):
            openai_ablation = cached_oa.get("ablation")

    if openai_ablation is None:
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("AZURE_OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set and cached openai_ablation_results.json not found.")
        with patch.object(ing, "EMBEDDING_MODEL_NAME", "text-embedding-3-small"):
            oa_live_ingestor = ing.ComplianceCorpusIngestor(storage_dir=openai_eval_dir)
            oa_live_ingestor.ingest(clean=False)
            oa_retriever = BitemporalComplianceRetriever(ingestor=oa_live_ingestor)
            openai_ablation = run_ablation_study(
                oa_retriever,
                eval_dataset["retrieval_queries"],
                modes=["dense_only", "dense_bitemporal", "hybrid_no_temporal", "hybrid_bm25", "hybrid_rerank"],
            )

    nomic_dense = ablation_results["nomic_dense_only"]
    qwen_dense = ablation_results["dense_only"]
    openai_dense = openai_ablation["dense_only"]

    # In pure 1-stage dense retrieval (no BM25, no bitemporal filter, no reranker),
    # OpenAI text-embedding-3-small outperforms both Qwen3-0.6B and Nomic-embed-text-v1.5
    assert openai_dense["assertion_pass_rate"] > qwen_dense["assertion_pass_rate"] >= nomic_dense["assertion_pass_rate"]
    assert openai_dense["hit_at_1"] > qwen_dense["hit_at_1"] >= nomic_dense["hit_at_1"]
    assert openai_dense["mrr_at_3"] > nomic_dense["mrr_at_3"]
    assert openai_dense["mrr_at_3"] > qwen_dense["mrr_at_3"]
    assert openai_dense["ndcg_at_3"] > nomic_dense["ndcg_at_3"] > qwen_dense["ndcg_at_3"]

    # With hard bitemporal filtering (dense_bitemporal), OpenAI also leads Qwen3
    assert openai_ablation["dense_bitemporal"]["assertion_pass_rate"] >= ablation_results["dense_bitemporal"]["assertion_pass_rate"]
    assert openai_ablation["dense_bitemporal"]["mrr_at_3"] > ablation_results["dense_bitemporal"]["mrr_at_3"]

    # In full 2-stage hybrid_rerank, both Qwen3 and OpenAI achieve >= 90% strict pass rate
    qwen_hr = ablation_results["hybrid_rerank"]
    openai_hr = openai_ablation["hybrid_rerank"]
    assert qwen_hr["assertion_pass_rate"] >= 90.0
    assert openai_hr["assertion_pass_rate"] >= 90.0
    assert qwen_hr["assertion_pass_rate"] >= openai_hr["assertion_pass_rate"]
    assert openai_hr["temporal_compliance_rate"] >= qwen_hr["temporal_compliance_rate"]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", str(Path(__file__).resolve()), *sys.argv[1:]]))
