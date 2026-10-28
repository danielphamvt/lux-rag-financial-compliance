"""Shared retrieval data structures, environment helpers, query routing, and declarative domain rules.

Used by `rag_engine.core` and all retrieval backends (`vector_chunking`, `page_index`)
so core modules never depend on backend implementations.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[2]
LOGS_DIR: Path = WORKSPACE_ROOT / "logs"

BPS_MULTIPLIER: Decimal = Decimal("10000")
AUDIT_RETRIEVAL_TOP_K: int = 3
AUDIT_CONFIDENCE_SCORE: float = 0.995
AUDIT_EXCERPT_MAX_CHARS: int = 400
NOMIC_SPAN_MIN_CHARS: int = 10
NOMIC_SPAN_MAX_CHARS: int = 450

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable, logging a warning and returning `default` on invalid values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid integer environment variable %s=%r; using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float environment variable, logging a warning and returning `default` on invalid values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid float environment variable %s=%r; using default %s", name, raw, default)
        return default


def validate_audit_date(audit_date: str) -> str:
    """Validate that `audit_date` is a strict `YYYY-MM-DD` ISO calendar date.

    Args:
        audit_date: Date string to validate.

    Returns:
        Cleaned `YYYY-MM-DD` string.

    Raises:
        ValueError: If `audit_date` is not a valid `YYYY-MM-DD` calendar date.
    """
    cleaned = audit_date.strip()
    if not _ISO_DATE_RE.match(cleaned):
        raise ValueError(f"Invalid audit_date format '{audit_date}'. Expected YYYY-MM-DD.")
    try:
        datetime.date.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(f"Invalid audit_date '{audit_date}'. Expected valid YYYY-MM-DD.") from exc
    return cleaned


def parse_date_to_int(date_str: str) -> int:
    """Validate ISO calendar date `YYYY-MM-DD` and convert into integer `YYYYMMDD`.

    Args:
        date_str: ISO calendar date string (`YYYY-MM-DD`).

    Returns:
        Integer representation `YYYYMMDD`.
    """
    cleaned = validate_audit_date(date_str)
    return int(cleaned.replace("-", ""))


@dataclass
class RetrievedParentChunk:
    """Single retrieved parent chunk or page node with bitemporal metadata and provenance digests."""

    parent_id: str
    doc_id: str
    source_file: str
    valid_from: str
    valid_to: str
    page_number: int
    section_id: str
    page_content: str
    rerank_score: float
    outbound_citations: list[str] = field(default_factory=list)
    sha256: str = ""
    md5: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """Standardized retrieval response shared across all backends and the NAV audit engine."""

    query: str
    audit_date: str | None
    parent_chunks: list[RetrievedParentChunk]
    mode: str = "hybrid_rerank"
    sub_queries: list[str] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        """Concatenate retrieved parent chunks with horizontal rule separators."""
        return "\n\n---\n\n".join(c.page_content for c in self.parent_chunks)

    @property
    def doc_ids(self) -> list[str]:
        """Deduplicated ordered list of document IDs across `parent_chunks`."""
        return list(dict.fromkeys(c.doc_id for c in self.parent_chunks))


COMPOUND_QUERY_SEPARATORS: tuple[str, ...] = (
    r",\s+and\s+(?:(?:for|to|in|under|by|from|on|at)\s+(?:[a-z0-9/-]+\s+){0,3},?\s*(?:what|which|who|whom|whose|where|when|how|does|do)|(?:what\s+(?:is|was|were)|which|who|whom|whose|where|when|how|does|do))\s+",
    r"\s+and\s+(?=(?:the|what|which)\s+[^?]*\b(?:table|section|article|chapter|point|paragraph)\s+\d+\b)",
)


def decompose_compound_query(
    query: str,
    separators: tuple[str, ...] = COMPOUND_QUERY_SEPARATORS,
) -> list[str]:
    """Split multi-part compliance queries into focused sub-queries."""
    q_clean = query.strip()
    for sep in separators:
        parts = re.split(sep, q_clean, maxsplit=1, flags=re.I)
        if len(parts) == 2:
            m = re.search(sep, q_clean, flags=re.I)
            conn = m.group(0).lstrip(",; ").strip() if m else ""
            prefix = conn[4:].strip() if conn.lower().startswith("and ") else conn
            return [parts[0].strip(), f"{prefix} {parts[1].strip()}".strip()]
    return [q_clean]


DOC_ROUTING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "LUX_ALPHA_PROSPECTUS": ("lux alpha", "prospectus", "tiered fee reduction", "fee schedule"),
    "UCITS_2009_65_EC": ("ucits directive", "directive 2009/65", "2009/65/ec"),
    "CSSF_24_856": ("cssf 24/856", "circular 24/856"),
    "CSSF_02_77": ("cssf 02/77", "circular 02/77"),
}


def infer_target_doc_id(
    query: str,
    routing_keywords: dict[str, tuple[str, ...]] = DOC_ROUTING_KEYWORDS,
) -> str | None:
    """Detect explicit statutory or fund document mentions in `query` from `routing_keywords`.

    Args:
        query: Input compliance query string.
        routing_keywords: Mapping of `doc_id` to lowercase routing keyword tuples.

    Returns:
        Matched `doc_id` or `None` when no document-specific keyword is present.
    """
    q_low = query.lower()
    return next((doc_id for doc_id, kws in routing_keywords.items() if any(k in q_low for k in kws)), None)


@dataclass(frozen=True)
class DomainConceptRule:
    """Declarative domain concept boost rule evaluated during Stage-2 reranking or tree navigation."""

    name: str
    weight: float
    query_any: tuple[str, ...]
    content_groups: tuple[tuple[str, ...], ...]
    section_any: tuple[str, ...] = ()

    def matches(self, q: str, s: str, c: str) -> bool:
        """Return True if lowercase query `q`, section `s`, and content `c` satisfy all rule groups."""
        if not any(k in q for k in self.query_any):
            return False
        for idx, grp in enumerate(self.content_groups):
            sec_terms = self.section_any if idx == 0 else ()
            if not (any(m in c for m in grp) or any(sm in s for sm in sec_terms)):
                return False
        return True
