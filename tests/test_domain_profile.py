"""Tests for the domain profile system.

The default profile (financial_compliance) must load cleanly, expose every key
the pipeline consumes, and render all prompt templates without format errors.
"""
import string

from domain_profile import DEFAULT_PROFILE, PROFILE, PROFILE_NAME, get_prompt

REQUIRED_SECTIONS = [
    "name", "response_language", "source_display_names", "source_keywords",
    "greetings", "small_talk",
    "internet_search", "markers", "strings", "prompts",
]

REQUIRED_PROMPTS = [
    "conversation_summary", "rewrite_query",
    "orchestrator", "stage1_retrieval_plan", "stage3_audit_synthesis",
    "orchestrator_knowledge_open", "orchestrator_sources_open",
    "orchestrator_knowledge_rag_only", "orchestrator_sources_rag_only",
    "fallback_response", "fallback_rules_open", "fallback_warning_open",
    "fallback_rules_rag_only", "fallback_warning_rag_only",
    "context_compression", "internet_search_answer",
    "internet_no_results_answer", "clarification_fallback",
]

REQUIRED_STRINGS = [
    "status_analyzing", "status_retrieving", "status_thinking",
    "status_answering", "status_processing", "error_notice",
    "retrieval_no_results", "forced_tool_empty_query", "out_of_scope_notice",
    "no_answers_generated", "unable_to_answer", "no_query_fallback",
    "internet_no_results", "internet_timeout", "internet_error",
]


def test_default_profile_loads_with_all_sections():
    assert PROFILE_NAME == "financial_compliance"
    for section in REQUIRED_SECTIONS:
        assert section in PROFILE and section in DEFAULT_PROFILE
    for key in REQUIRED_STRINGS:
        assert key in PROFILE["strings"] and key in DEFAULT_PROFILE["strings"]


def test_marker_configuration_is_consistent():
    markers = PROFILE["markers"]
    assert markers["reasoning_primary"] in markers["reasoning"]
    assert markers["conclusion_primary"] in markers["conclusion"]
    assert markers["reasoning_closing"].strip()


def test_get_prompt_renders_with_runtime_fields():
    # Every template renders with sentinel values for its placeholders
    # (markers + language are injected by the loader itself).
    sentinel = {
        "context": "CTX",
        "question": "Q",
        "hist_section": "HIST",
        "knowledge_rule": "KR",
        "sources_fallback": "SF",
        "source_integrity_rule": "SIR",
        "sources_warning": "SW",
        "tool_name_to_force": "search_documents",
    }
    for key in REQUIRED_PROMPTS:
        template = PROFILE["prompts"][key]
        needed = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        assert get_prompt(key, **{k: v for k, v in sentinel.items() if k in needed}) != ""


def test_missing_profile_falls_back_to_generic(tmp_path, monkeypatch):
    import domain_profile
    monkeypatch.setattr(domain_profile, "PROFILE_NAME", "does_not_exist")
    monkeypatch.setattr(domain_profile, "PROFILES_DIR", tmp_path)
    profile = domain_profile._load_profile()
    assert profile["name"] == "generic"


def test_partials_are_expanded_at_load_time():
    """Verify all {{_partial}} placeholders are expanded when PROFILE is loaded."""
    for key, template in PROFILE["prompts"].items():
        assert "{{_" not in template, f"Unexpanded partial found in prompt '{key}'"
    # Verify shared partial content was actually injected into consuming prompts
    assert "MULTI-TIMELINE BITEMPORAL GOVERNANCE" in PROFILE["prompts"]["orchestrator"]
    assert "SYMMETRICAL MULTI-TARGET" in PROFILE["prompts"]["stage3_audit_synthesis"]
    assert "SYMMETRICAL MULTI-TARGET" in PROFILE["prompts"]["fallback_response"]


def test_safe_runtime_render_ignores_latex_and_json_braces(monkeypatch):
    """Verify get_prompt() safely preserves unknown {braces} (e.g. LaTeX or JSON) without KeyError."""
    import domain_profile
    custom_prompts = dict(domain_profile.PROFILE["prompts"])
    custom_prompts["latex_test"] = (
        r"Formula: $$\frac{a}{b}$$ and JSON: {'key': 'val'} in {response_language} with {reasoning_primary}."
    )
    monkeypatch.setitem(domain_profile.PROFILE, "prompts", custom_prompts)
    rendered = domain_profile.get_prompt("latex_test")
    assert r"\frac{a}{b}" in rendered
    assert "{'key': 'val'}" in rendered
    assert domain_profile.PROFILE["response_language"] in rendered
    assert domain_profile.PROFILE["markers"]["reasoning_primary"] in rendered


def test_progressive_disclosure_turn1_token_budget():
    """Turn 1 instructions (orchestrator + orchestrator_plan) must stay <= 600 tokens with >= 35 tokens headroom."""
    import tiktoken
    from rag_agent.prompts import get_orchestrator_plan_prompt, get_orchestrator_prompt

    enc = tiktoken.get_encoding("cl100k_base")
    for rag_only in (True, False):
        sys_tokens = len(enc.encode(get_orchestrator_prompt(rag_only=rag_only)))
        for forced_tool in ("search_documents", "run_nav_audit"):
            plan_tokens = len(enc.encode(get_orchestrator_plan_prompt(forced_tool=forced_tool)))
            total_turn1_tokens = sys_tokens + plan_tokens
            assert total_turn1_tokens <= 565, (
                f"Turn 1 token budget exceeded (rag_only={rag_only}, tool={forced_tool}): {total_turn1_tokens} > 565"
            )


def test_claude_code_governance_patterns_and_synthesis_token_budget():
    """Verify the 5 Claude Code governance patterns and compacted Turn 2+ synthesis token budget (<= 2250)."""
    import tiktoken
    from rag_agent.prompts import (
        get_orchestrator_plan_prompt,
        get_orchestrator_prompt,
        get_orchestrator_synthesis_prompt,
        get_rewrite_query_prompt,
    )

    enc = tiktoken.get_encoding("cl100k_base")
    sys_prompt = get_orchestrator_prompt(rag_only=True)
    plan_prompt = get_orchestrator_plan_prompt(forced_tool="search_documents")
    synth_rag = get_orchestrator_synthesis_prompt(rag_only=True)
    synth_open = get_orchestrator_synthesis_prompt(rag_only=False)
    rewrite_prompt = get_rewrite_query_prompt()

    # 0. Immutable Top-Level System Directive
    from rag_agent.prompts import IMMUTABLE_SYSTEM_DIRECTIVE
    assert IMMUTABLE_SYSTEM_DIRECTIVE in sys_prompt
    assert IMMUTABLE_SYSTEM_DIRECTIVE in get_orchestrator_prompt(rag_only=False)
    assert "Be direct, brief, and concise." in sys_prompt
    assert "If you're unsure about the user's intent, ask for clarification rather than making assumptions." in sys_prompt
    # 1. External Payload Trust Boundary
    assert "EXTERNAL PAYLOAD TRUST BOUNDARY" in sys_prompt
    # 2. 3-State Regulatory Verification
    assert "3-STATE VERIFICATION" in synth_rag
    assert "COMPLIANT / EXEMPT" in synth_rag
    assert "DEFINITE BREACH / MATERIAL ERROR" in synth_rag
    assert "INDETERMINATE / ABSTAIN" in synth_rag
    # 3. 4-Part Finding-to-Remediation Schema & Terminal Sources Footer
    assert "FINDING-TO-REMEDIATION SCHEMA" in synth_rag
    assert "TERMINAL SOURCES FOOTER" in synth_rag
    assert "strictly BEFORE `---` and `**SOURCES**` (never after `**SOURCES**`)" in synth_open
    assert "Immediately below `### CONCLUSION`" in synth_open
    assert "⚠️ **WARNING — Data Source:" in synth_open
    assert "INDETERMINATE — Missing Required Audit Inputs" in synth_rag
    assert PROFILE["internet_search"]["query_suffixes"] == []
    # 4. Complete Tool Provenance & Predicate-First Filtering
    assert "Homogeneous Units & Predicate-First Filtering" in synth_rag
    assert "Economic Sign & Cost Symmetry" in synth_rag
    assert "VERBATIM TOOL PROVENANCE" in synth_rag
    # 5. Autonomous Clarification Bias
    assert "CLARIFICATION BIAS & TEMPORAL FILTER" in plan_prompt
    assert "Autonomous Clarification Bias" in rewrite_prompt

    # Synthesis token compaction budget (<= 2250 tokens)
    assert len(enc.encode(synth_rag)) <= 2250
    assert len(enc.encode(synth_open)) <= 2250


def test_streaming_markers_isolated_to_orchestrator():
    """Only orchestrator prompts should emit [REASONING]/[CONCLUSION]; secondary nodes stream directly to final."""
    from rag_agent.prompts import (
        get_fallback_response_prompt,
        get_orchestrator_synthesis_prompt,
    )

    r_marker = PROFILE["markers"]["reasoning_primary"]
    c_marker = PROFILE["markers"]["conclusion_primary"]

    synth = get_orchestrator_synthesis_prompt(rag_only=True)
    assert r_marker in synth
    assert c_marker in synth

    for secondary in [
        get_fallback_response_prompt(rag_only=True),
        get_fallback_response_prompt(rag_only=False),
    ]:
        assert r_marker not in secondary
        assert c_marker not in secondary


def test_math_partial_forbids_latex_and_sanitizer_cleans_broken_math():
    """Verify all synthesis prompts forbid LaTeX macros and clean_display_sources / content_to_text normalize residual LaTeX."""
    from ui.renderers import clean_display_sources
    from rag_agent.nodes_edges import content_to_text
    from rag_agent.prompts import (
        get_context_compression_prompt,
        get_fallback_response_prompt,
        get_orchestrator_prompt,
        get_orchestrator_synthesis_prompt,
    )

    # Default prompt builders must enforce strict RAG-only (Closed-World) mode with zero internet references
    for default_prompt in [
        get_orchestrator_prompt(),
        get_orchestrator_synthesis_prompt(),
        get_fallback_response_prompt(),
    ]:
        assert "ABSOLUTELY FORBIDDEN" in default_prompt or "CLOSED-WORLD ABSTENTION SOURCES FOOTER" in default_prompt
        assert "[INTERNET_DATA_START]" not in default_prompt
        assert "EXTERNAL PAYLOAD TRUST BOUNDARY" in default_prompt or "VERBATIM TOOL PROVENANCE" in default_prompt

    for prompt_text in [
        get_orchestrator_synthesis_prompt(rag_only=True),
        get_fallback_response_prompt(rag_only=True),
    ]:
        assert "STRICT ZERO-LATEX RULE" in prompt_text
        assert "FORBIDDEN (breaks UI):" in prompt_text
        assert "REQUIRED (clean plain-text Unicode):" in prompt_text
        assert "25,000,000 × 0.0075 × (30 / 365) = 15,410.96 EUR" in prompt_text
        assert "(12,500,000 / 200,000,000) × 100 = 6.25%" in prompt_text
        # Ensure benchmark scenario-specific answers (S03/S06/S07/CALC_02/CALC_13) are not hardcoded in prompts
        assert "4,109.59" not in prompt_text
        assert "48,000,000 / 500,000,000" not in prompt_text
        assert "49,863.01" not in prompt_text
        assert "650 bps > 500 bps" not in prompt_text
        assert "Caisse de consignation" not in prompt_text

    assert "NEVER use LaTeX macros" in get_context_compression_prompt()

    raw_latex = (
        r"Tier 1: \( 50,000,000 \times 0.004 \times \frac{91}{365} = 49,863.01 \, \text{EUR} \)" "\n"
        r"Tier 2: ( 10,000,000 \times 0.003 \times \frac{91}{365} = 7,479.45 , \text{EUR} )"
    )
    cleaned = clean_display_sources(raw_latex)
    assert cleaned == (
        "Tier 1: 50,000,000 × 0.004 × (91 / 365) = 49,863.01 EUR\n"
        "Tier 2: 10,000,000 × 0.003 × (91 / 365) = 7,479.45 EUR"
    )
    assert content_to_text(r"\( \frac{48,000,000}{500,000,000} \times 100 = 9.60\% \)") == "(48,000,000 / 500,000,000) × 100 = 9.60%"


def test_calculation_techniques_coordinator_activates_on_demand():
    """Verify 02_calculation_techniques/coordinator.md and 3-stage progressive disclosure."""
    import tiktoken
    from domain_profile import PROFILE, select_calculation_techniques
    from rag_agent.prompts import (
        get_orchestrator_calculation_gate_prompt,
        get_orchestrator_synthesis_prompt,
    )

    enc = tiktoken.get_encoding("cl100k_base")
    loaded_files = [t["file"] for t in PROFILE.get("calculation_techniques", [])]
    assert len(loaded_files) == 5
    assert "breach-indemnification-accounting-vs-economic.md" in loaded_files
    assert "swing-pricing-and-anti-dilution.md" in loaded_files

    assert select_calculation_techniques("") == ""
    indemn_gate = get_orchestrator_calculation_gate_prompt(
        query_context="Audit indemnification under the Accounting Method vs Economic Method for realized loss.",
    )
    assert "PHASE 2A" in indemn_gate
    assert "ACTIVE TECHNIQUE — Investment Breach Indemnification" in indemn_gate
    assert "ACTIVE TECHNIQUE — Swing Pricing" not in indemn_gate
    assert "STEP 2 — FULL AUDIT-GRADE SYNTHESIS" not in indemn_gate
    assert len(enc.encode(indemn_gate)) <= 900

    swing_gate = get_orchestrator_calculation_gate_prompt(
        query_context="Compute swung NAV and dilution adjustment for subscriptions and redemptions.",
    )
    assert "ACTIVE TECHNIQUE — Swing Pricing & Anti-Dilution Levy" in swing_gate
    assert "ACTIVE TECHNIQUE — Investment Breach Indemnification" not in swing_gate

    phase2b_synth = get_orchestrator_synthesis_prompt(
        rag_only=True,
        query_context="Compute swung NAV and dilution adjustment.",
        include_calc_gate=False,
    )
    assert "PHASE 2B" in phase2b_synth
    assert "ACTIVE TECHNIQUE" not in phase2b_synth
    assert len(enc.encode(phase2b_synth)) <= 1800




