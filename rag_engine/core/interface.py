"""Backend contract: one implementation, two consumers (UI + evaluation)."""
from __future__ import annotations

from typing import Protocol

from rag_engine.core.schemas import RetrievalResult


class ComplianceBackend(Protocol):
    """Minimal contract for one retrieval architecture.

    retrieve() is deterministic: same index + query + audit_date -> same result.
    No LLM calls or hidden state inside retrieval; conversation logic
    (rewrite/routing/history) lives in the frontend, outside this contract.
    """

    name: str
    min_score: float
    requires_warmup: bool

    def ingest(self) -> object: ...

    def retrieve(self, query: str, audit_date: str | None = None, top_k: int = 5) -> RetrievalResult: ...

    def warm_up(self) -> bool: ...

