"""Pre-execution quantitative query intent detector & tool-call gate (`quantitative_query_gate.py`).

Detects whether a user question requires numerical calculation and verifies whether
a deterministic symbolic math tool (`calculate_decimal` or `run_nav_audit`) has been executed.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

_QUANT_NUMBERS_RE = re.compile(
    r"\b\d{2,}[\d,._]*\b|\b\d+(?:\.\d+)?\s*%|\b\d+\s*(?:bps|issuers?|classes|days?|months?|years?|tiers?|tranches?|holdings?|positions?)\b",
    re.IGNORECASE,
)
_QUANT_INTENT_RE = re.compile(
    r"\b(?:calculate|compute|quantify|how\s+much|how\s+many|total|subtotal|difference|variance|"
    r"overstatement|understatement|ratio|percentage|exposure|accrual|accrued|rate|amount|exceed|breach)\b",
    re.IGNORECASE,
)


def has_symbolic_math_tool_call(agent_messages: list | None, ledger: list | None = None) -> bool:
    """Return True if a valid `calculate_decimal` or non-rejected `run_nav_audit` call ran in the current turn.

    Args:
        agent_messages: Sequence of LangChain messages in the active sub-agent turn.
        ledger: Optional durable `ToolExecutionRecord` sequence from graph state.

    Returns:
        True if at least one non-rejected symbolic math tool call is present, False otherwise.
    """
    rejected_tc_ids = {
        getattr(m, "tool_call_id", None)
        for m in (agent_messages or [])
        if isinstance(m, ToolMessage) and str(getattr(m, "content", "")).startswith("[SCOPE_GUARD:")
    }
    for msg in agent_messages or []:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.get("name") in ("calculate_decimal", "run_nav_audit") and tc.get("id") not in rejected_tc_ids:
                    return True
    for r in ledger or []:
        if getattr(r, "tool_name", "") in ("calculate_decimal", "run_nav_audit"):
            out_s = str(getattr(r, "output", "") or getattr(r, "output_first_line", "") or "")
            if not out_s.startswith("[SCOPE_GUARD:") and getattr(r, "tool_call_id", None) not in rejected_tc_ids:
                return True
    return False


def is_quantitative_query(question: str) -> bool:
    """Domain-agnostic check for whether a query requires numerical computation.

    Args:
        question: Raw or rewritten user query string.

    Returns:
        True if the query contains both numerical figures/units and quantitative intent keywords.
    """
    if not question:
        return False
    return bool(_QUANT_NUMBERS_RE.search(question) and _QUANT_INTENT_RE.search(question))
