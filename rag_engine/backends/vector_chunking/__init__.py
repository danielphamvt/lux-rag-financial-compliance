"""vector_chunking: bitemporal hybrid retrieval over table-aware parent-child chunking.

Registered in rag_engine/core/registry.py. Conversation logic stays in the frontend;
this package exposes only ingest + deterministic retrieve.
"""
from __future__ import annotations

from rag_engine.core.schemas import RetrievalResult, RetrievedChunk

from .vector_chunking_config import (
    DEFAULT_TOP_K,
    VECTOR_CHUNKING_CONFIG,
    VectorChunkingConfig,
)


class VectorChunkingBackend:
    """Thin adapter mapping the engine onto the shared backend contract.

    The concrete engine objects (.ingestor / .retriever) stay accessible for the
    benchmark's mode-level ablation studies; the UI and conformance suite go
    through ingest()/retrieve() only.
    """

    def __init__(self, config: VectorChunkingConfig = VECTOR_CHUNKING_CONFIG) -> None:
        self.config: VectorChunkingConfig = config
        self.name: str = config.backend_name
        self.min_score: float = config.min_relevance_score
        self.requires_warmup: bool = config.requires_warmup
        self._ingestor = None
        self._retriever = None

    def _get_ingestor(self):
        if self._ingestor is None:
            from .ingest import ComplianceCorpusIngestor

            self._ingestor = ComplianceCorpusIngestor(
                data_docs_dir=self.config.data_docs_dir,
                storage_dir=self.config.storage_dir,
                collection_name=self.config.collection_name,
            )
        return self._ingestor

    @property
    def ingestor(self):
        """Concrete engine ingestor (the eval path needs the manifest and clean re-ingest)."""
        return self._get_ingestor()

    def ingest(self) -> dict:
        """Contract entry: ensure the index exists (idempotent) and return the manifest."""
        return self._get_ingestor().ingest(clean=False)

    @property
    def retriever(self):
        """Concrete engine retriever (the benchmark ablation studies use it directly)."""
        if self._retriever is None:
            from .retrieval import BitemporalComplianceRetriever

            self._retriever = BitemporalComplianceRetriever(ingestor=self.ingestor)
        return self._retriever

    def warm_up(self) -> bool:
        """Warm up the backend's models and index with one sample retrieval."""
        if not self.requires_warmup:
            return False
        self.retrieve(
            self.config.warmup_query,
            audit_date=self.config.warmup_audit_date,
            top_k=self.config.warmup_top_k,
        )
        return True

    def retrieve(self, query: str, audit_date: str | None = None, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        """Contract entry: deterministic retrieval mapped onto the shared result schema."""
        resp = self.retriever.retrieve(
            query=query,
            audit_date=audit_date,
            mode=self.config.default_retrieval_mode,
            top_k=top_k,
        )
        return self._to_result(resp)

    @staticmethod
    def _to_result(resp) -> RetrievalResult:
        chunks = [
            RetrievedChunk(
                id=c.parent_id,
                text=c.page_content,
                score=c.rerank_score,
                doc_id=c.doc_id,
                source_file=c.source_file,
                section_id=c.section_id,
                page_number=c.page_number,
                valid_from=c.valid_from,
                valid_to=c.valid_to,
            )
            for c in resp.parent_chunks
        ]
        return RetrievalResult(query=resp.query, audit_date=resp.audit_date, chunks=chunks)


def create_backend() -> VectorChunkingBackend:
    return VectorChunkingBackend()
