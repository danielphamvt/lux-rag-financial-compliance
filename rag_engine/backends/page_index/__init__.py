"""page_index: SOTA vectorless page-level & document-tree navigation RAG backend.

Registered in rag_engine/core/registry.py. Preserves full pages, Markdown tables,
and footnotes atomically in `data/indexes/page_index/document_tree.json`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rag_engine.core.interface import ComplianceBackend
from rag_engine.core.schemas import RetrievalResult, RetrievedChunk

from .page_index_config import (
    DEFAULT_TOP_K,
    PAGE_INDEX_CONFIG,
    PageIndexConfig,
)

logger = logging.getLogger(__name__)


class PageIndexBackend(ComplianceBackend):
    """Adapter mapping PageIndexIngestor and PageIndexRetriever onto ComplianceBackend."""

    def __init__(self, config: PageIndexConfig = PAGE_INDEX_CONFIG) -> None:
        self.config: PageIndexConfig = config
        self.name: str = config.backend_name
        self.min_score: float = config.min_relevance_score
        self.requires_warmup: bool = config.requires_warmup
        self._ingestor = None
        self._retriever = None

    def _get_ingestor(self, corpus_dir: str | Path | None = None):
        target_dir = Path(corpus_dir) if corpus_dir is not None else self.config.data_docs_dir
        if self._ingestor is None or self._ingestor.data_docs_dir != target_dir:
            from .ingest import PageIndexIngestor

            self._ingestor = PageIndexIngestor(
                data_docs_dir=target_dir,
                index_dir=self.config.index_dir,
            )
            self._retriever = None
        return self._ingestor

    @property
    def ingestor(self):
        return self._get_ingestor()

    def ingest(self, corpus_dir: str | Path | None = None) -> dict:
        return self._get_ingestor(corpus_dir).ingest(clean=False)

    @property
    def retriever(self):
        if self._retriever is None:
            from .retrieval import PageIndexRetriever

            self._retriever = PageIndexRetriever(ingestor=self.ingestor)
        return self._retriever

    def warm_up(self) -> bool:
        """Vectorless tree-based backend requires no neural model warm-up."""
        if not self.requires_warmup:
            logger.info("Backend '%s' (vectorless, tree-based) does not require model warm-up.", self.name)
            return False
        return True

    def retrieve(self, query: str, audit_date: str | None = None, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        resp = self.retriever.retrieve(query=query, audit_date=audit_date, top_k=top_k)
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


def create_backend() -> PageIndexBackend:
    """Create the default `PageIndexBackend` instance."""
    return PageIndexBackend()
