"""Fixed result schemas shared by every backend (part of the backend contract)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """One retrieved parent chunk.

    Optional fields are None when a backend does not provide them
    (eval reports N/A instead of guessing).
    """

    id: str  # unique chunk id (e.g. parent_id)
    text: str  # chunk content
    score: float  # ranking score
    doc_id: str | None = None
    source_file: str | None = None
    section_id: str | None = None
    page_number: int | None = None
    valid_from: str | None = None
    valid_to: str | None = None


@dataclass
class RetrievalResult:
    """Fixed result schema every backend returns (eval metrics run on this unchanged)."""

    query: str
    audit_date: str | None
    chunks: list[RetrievedChunk] = field(default_factory=list)
