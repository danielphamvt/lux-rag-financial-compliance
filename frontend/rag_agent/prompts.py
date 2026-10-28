"""Prompt builders.

All prompt wording lives in the active domain profile (see
``domain_profile.py`` and ``profiles/<name>/profile.json``). This module only
selects structural variants (open vs RAG-only) and renders templates with
their runtime fields.
"""

from domain_profile import _DEFAULT_DIRECTIVE, PROFILE, get_prompt, select_calculation_techniques

IMMUTABLE_SYSTEM_DIRECTIVE = (
    PROFILE.get("partials", {}).get("_immutable_directive")
    or _DEFAULT_DIRECTIVE
)


def _enforce_immutable_system_directive(prompt: str) -> str:
    """Guarantee the immutable system directive is present in SystemMessage[0] at the end."""
    directive = PROFILE.get("partials", {}).get("_immutable_directive") or IMMUTABLE_SYSTEM_DIRECTIVE
    if directive in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n{directive}"


def get_conversation_summary_prompt() -> str:
    return get_prompt("conversation_summary")


def get_rewrite_query_prompt() -> str:
    return get_prompt("rewrite_query")


def get_orchestrator_prompt(rag_only: bool = True) -> str:
    variant = "rag_only" if rag_only else "open"
    knowledge_rule = get_prompt(f"orchestrator_knowledge_{variant}")
    rendered = get_prompt(
        "orchestrator",
        knowledge_rule=knowledge_rule,
    )
    return _enforce_immutable_system_directive(rendered)


def get_orchestrator_plan_prompt(forced_tool: str | None = None) -> str:
    tool_name_to_force = forced_tool if forced_tool else "search_documents"
    return get_prompt(
        "stage1_retrieval_plan",
        tool_name_to_force=tool_name_to_force,
    )


def get_orchestrator_calculation_gate_prompt(query_context: str = "") -> str:
    """Render Phase 2A Calculation Gate prompt with only the active calculation technique."""
    active_techniques = select_calculation_techniques(query_context)
    return get_prompt(
        "stage2_calculation_gate",
        active_calculation_techniques=active_techniques,
    )


def get_orchestrator_synthesis_prompt(
    rag_only: bool = True,
    query_context: str = "",
    include_calc_gate: bool = True,
) -> str:
    variant = "rag_only" if rag_only else "open"
    sources_fallback = get_prompt(f"orchestrator_sources_{variant}")
    fields = {"sources_fallback": sources_fallback}
    if include_calc_gate:
        fields["active_calculation_techniques"] = select_calculation_techniques(query_context)
    else:
        fields["calculation_gate_section"] = PROFILE.get("partials", {}).get("_compact_calc_verification", "")
        fields["active_calculation_techniques"] = ""
    return get_prompt(
        "stage3_audit_synthesis",
        **fields,
    )


def get_fallback_response_prompt(rag_only: bool = True) -> str:
    variant = "rag_only" if rag_only else "open"
    source_integrity_rule = get_prompt(f"fallback_rules_{variant}")
    sources_warning = get_prompt(f"fallback_warning_{variant}")
    return get_prompt(
        "fallback_response",
        source_integrity_rule=source_integrity_rule,
        sources_warning=sources_warning,
    )


def get_context_compression_prompt() -> str:
    return get_prompt("context_compression")


def get_internet_search_prompt(context: str, question: str, history_context: str = "") -> str:
    hist_section = (
        f"\nConversation Context (for reference):\n{history_context.strip()}\n"
        if history_context.strip() else ""
    )
    return get_prompt(
        "internet_search_answer",
        hist_section=hist_section,
        context=context,
        question=question,
    )


def get_internet_no_results_prompt(question: str) -> str:
    return get_prompt("internet_no_results_answer", question=question)
