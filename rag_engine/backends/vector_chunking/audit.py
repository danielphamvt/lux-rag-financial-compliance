"""Backward-compatible facade re-exporting the Neuro-Symbolic NAV Audit Engine and schemas.

All core schemas and `NeuroSymbolicNAVAuditEngine` live in `rag_engine.core.nav_audit_schemas`
and `rag_engine.core.nav_audit` so both the frontend agent and evaluation suite depend on
`rag_engine.core` rather than reaching inside `rag_engine.backends.vector_chunking`.
"""

from __future__ import annotations

from rag_engine.core.nav_audit import NAVAuditGraphState, NeuroSymbolicNAVAuditEngine
from rag_engine.core.nav_audit_parsers import (
    parse_prospectus_fee_from_chunks,
    parse_statutory_threshold_from_chunks,
)
from rag_engine.core.nav_audit_schemas import (
    FundAccountingPackInput,
    NAVAuditResult,
    ProspectusFeeSpec,
    StatutoryThresholdSpec,
    round_bps,
    round_currency,
)

__all__ = [
    "FundAccountingPackInput",
    "NAVAuditGraphState",
    "NAVAuditResult",
    "NeuroSymbolicNAVAuditEngine",
    "ProspectusFeeSpec",
    "StatutoryThresholdSpec",
    "parse_prospectus_fee_from_chunks",
    "parse_statutory_threshold_from_chunks",
    "round_bps",
    "round_currency",
]
