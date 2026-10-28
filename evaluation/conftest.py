"""
Shared pytest fixtures and helpers for evaluation test suites.

Provides session-scoped fixtures:
- `ingestor`: Instance of ComplianceCorpusIngestor with corpus indexed.
- `retriever`: Instance of BitemporalComplianceRetriever.
- `audit_engine`: Instance of NeuroSymbolicNAVAuditEngine.
- `eval_dataset`: Full evaluation dataset dict (100 retrieval queries).
- `ablation_results`: Persisted or computed ablation study metrics.
- `scenario_results`: Persisted or computed NAV audit scenario results.
- `load_full_queries`: Helper to load 100 retrieval queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluation.eval_common import WORKSPACE_ROOT, ensure_runtime

# Strictly enforce isolated test environment (.env.test) for all evaluation suites
ensure_runtime()

import pytest

if TYPE_CHECKING:
    from rag_engine.backends.vector_chunking.audit import NeuroSymbolicNAVAuditEngine
    from rag_engine.backends.vector_chunking.ingest import ComplianceCorpusIngestor
    from rag_engine.backends.vector_chunking.retrieval import BitemporalComplianceRetriever

LOGS_DIR = WORKSPACE_ROOT / "logs"
BENCHMARK_RESULTS_PATH = (
    WORKSPACE_ROOT / "evaluation" / "results" / "vector_chunking" / "benchmark_results.json"
)


def load_full_queries() -> list[dict[str, Any]]:
    """Helper nạp toàn bộ 100 câu retrieval queries (40 baseline + 60 stress).

    Returns:
        List of 100 normalized query specification dictionaries.
    """
    from evaluation.run_benchmark import load_full_evaluation_dataset

    dataset = load_full_evaluation_dataset()
    return dataset["retrieval_queries"]


@pytest.fixture(scope="session")
def ingestor() -> ComplianceCorpusIngestor:
    """Session-scoped fixture providing an indexed ComplianceCorpusIngestor."""
    from rag_engine.backends.vector_chunking.ingest import ComplianceCorpusIngestor

    ing = ComplianceCorpusIngestor()
    ing.ingest(clean=False)
    return ing


@pytest.fixture(scope="session")
def retriever(ingestor: ComplianceCorpusIngestor) -> BitemporalComplianceRetriever:
    """Session-scoped fixture providing BitemporalComplianceRetriever."""
    from rag_engine.backends.vector_chunking.retrieval import BitemporalComplianceRetriever

    return BitemporalComplianceRetriever(ingestor=ingestor)


@pytest.fixture(scope="session")
def audit_engine(retriever: BitemporalComplianceRetriever) -> NeuroSymbolicNAVAuditEngine:
    """Session-scoped fixture providing NeuroSymbolicNAVAuditEngine."""
    from rag_engine.backends.vector_chunking.audit import NeuroSymbolicNAVAuditEngine

    return NeuroSymbolicNAVAuditEngine(retriever=retriever)


@pytest.fixture(scope="session")
def eval_dataset() -> dict[str, Any]:
    """Session-scoped fixture providing the full 100-query evaluation dataset."""
    from evaluation.run_benchmark import load_full_evaluation_dataset

    return load_full_evaluation_dataset()


@pytest.fixture(scope="session")
def ablation_results(retriever: BitemporalComplianceRetriever, eval_dataset: dict[str, Any]) -> dict[str, Any]:
    """Session-scoped fixture providing 6-mode retrieval ablation study metrics.

    Reuses cached benchmark_results.json if valid and version-matched;
    otherwise falls back to live execution.
    """
    from evaluation.run_benchmark import ALL_ABLATION_MODES, run_ablation_study

    if BENCHMARK_RESULTS_PATH.exists():
        cached = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
        ab = cached.get("ablation", {})
        if (
            cached.get("benchmark_version") == eval_dataset.get("metadata", {}).get("version", "4.0.0")
            and set(ab.keys()) == set(ALL_ABLATION_MODES)
            and ab.get("hybrid_rerank", {}).get("total_queries") == len(eval_dataset["retrieval_queries"])
        ):
            return ab
    return run_ablation_study(retriever, eval_dataset["retrieval_queries"], modes=ALL_ABLATION_MODES)


@pytest.fixture(scope="session")
def scenario_results(audit_engine: NeuroSymbolicNAVAuditEngine) -> dict[str, Any]:
    """Session-scoped fixture providing results across all 8 NAV audit scenarios (10 runs)."""
    from evaluation.run_benchmark import run_all_benchmark_scenarios

    return run_all_benchmark_scenarios(audit_engine)
