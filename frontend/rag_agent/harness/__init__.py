"""Deterministic Agent Harness (`frontend/rag_agent/harness/`) for the Financial Compliance RAG Agent.

Four Defense-in-Depth stages, one module each (see `README_HARNESS.md`):
- `quantitative_query_gate` / `multi_turn_calc_continuity` / `nav_audit_contract_guard`: pre-execution gates.
- `ast_decimal_evaluator` / `portfolio_aggregation_engine`: safe symbolic `Decimal` (`ROUND_HALF_UP`) compute.
- `latex_to_unicode_normalizer` / `citation_and_warning_guard`: output & citation verification.
- `response_protocol_enforcer`: 5-marker `[REASONING]` / `[CONCLUSION]` protocol enforcer and UI lane splitter.
"""

from rag_agent.harness.ast_decimal_evaluator import (
    collect_top_level_addends,
    eval_decimal_node,
    normalize_calc_expression,
)
from rag_agent.harness.citation_and_warning_guard import (
    AI_DISCLAIMER,
    extract_fallback_sources_footer,
    is_closed_abstention,
    normalize_external_source_warning,
    retrieved_high_rel_files,
    strip_conditional_disclaimer,
    verify_sources_footer,
)
from rag_agent.harness.latex_to_unicode_normalizer import sanitize_plain_text_math
from rag_agent.harness.multi_turn_calc_continuity import (
    build_compressed_record_messages,
    merge_prior_calc_legs,
    split_label_and_expr,
)
from rag_agent.harness.nav_audit_contract_guard import (
    format_nav_audit_output,
    validate_nav_audit_contract,
)
from rag_agent.harness.portfolio_aggregation_engine import evaluate_decimal_expressions
from rag_agent.harness.quantitative_query_gate import (
    has_symbolic_math_tool_call,
    is_quantitative_query,
)
from rag_agent.harness.response_protocol_enforcer import (
    enforce_response_protocol,
    normalize_protocol_headings,
    split_reasoning_and_conclusion,
)

__all__ = [
    "AI_DISCLAIMER",
    "build_compressed_record_messages",
    "collect_top_level_addends",
    "enforce_response_protocol",
    "eval_decimal_node",
    "evaluate_decimal_expressions",
    "extract_fallback_sources_footer",
    "format_nav_audit_output",
    "has_symbolic_math_tool_call",
    "is_closed_abstention",
    "is_quantitative_query",
    "merge_prior_calc_legs",
    "normalize_calc_expression",
    "normalize_external_source_warning",
    "normalize_protocol_headings",
    "retrieved_high_rel_files",
    "sanitize_plain_text_math",
    "split_label_and_expr",
    "split_reasoning_and_conclusion",
    "strip_conditional_disclaimer",
    "validate_nav_audit_contract",
    "verify_sources_footer",
]
