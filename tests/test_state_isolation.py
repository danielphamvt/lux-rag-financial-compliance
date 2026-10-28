"""Regression tests for agent state isolation.

Guards the fix that gives the agent a PRIVATE `agent_messages` channel so its
internal reasoning never leaks into the parent graph's user-visible `messages`. These
tests are LLM-free (no Ollama required) — they assert structure and pure-function behavior.
"""
from langchain_core.messages import AIMessage

from rag_agent.graph import agent_graph
from rag_agent.nodes_edges import aggregate_answers, content_to_text
from rag_agent.state import accumulate_or_reset


def test_graph_has_separate_channels():
    """Verify that both messages and agent_messages channels exist in the main graph and are separate."""
    channels = set(agent_graph.channels.keys())
    assert "messages" in channels, "graph is missing main `messages` channel"
    assert "agent_messages" in channels, "graph is missing private `agent_messages` channel"


def test_accumulate_or_reset_resets_on_sentinel():
    """The per-turn reset sentinel clears accumulated answers."""
    existing = [{"index": 0, "answer": "old"}]
    assert accumulate_or_reset(existing, [{"__reset__": True}]) == []
    # Normal accumulation still appends.
    assert accumulate_or_reset(existing, [{"index": 1, "answer": "new"}]) == [
        {"index": 0, "answer": "old"},
        {"index": 1, "answer": "new"},
    ]


def test_aggregate_single_answer_skips_llm():
    """One sub-query returns the answer verbatim without an LLM synthesis call."""
    state = {"agent_answers": [{"index": 0, "question": "q", "answer": "direct answer"}], "originalQuery": "q"}
    result = aggregate_answers(state)
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "direct answer"


def test_aggregate_empty_answers_guard():
    """No answers → graceful message, no crash, no LLM call."""
    result = aggregate_answers({"agent_answers": [], "originalQuery": "q"})
    assert "No answers were generated." in result["messages"][0].content


def test_content_to_text_coerces_block_lists():
    """Provider content can arrive as a block list; coerce to plain text."""
    assert content_to_text("plain") == "plain"
    assert content_to_text([{"type": "text", "text": "a"}, {"type": "tool_use"}, "b"]) == "ab"
    assert content_to_text([]) == ""
    assert content_to_text(None) == ""


def test_dual_regime_date_inference_does_not_filter_out_old_regime():
    """Queries spanning 2024/02-77 and 2025/24-856 return None so both regimes remain searchable."""
    from rag_agent.tools import _infer_audit_date_from_query
    assert _infer_audit_date_from_query("Compare CSSF 02/77 (2024) and CSSF 24/856 (2025)") is None
    assert _infer_audit_date_from_query("Error occurred in 2024 and detected in 2025") is None
    assert _infer_audit_date_from_query("As of 2024-06-15, what is the MMF threshold?") == "2024-06-15"
    assert _infer_audit_date_from_query("Which circular applies between 29 March 2024 and 1 January 2025 in Chapter 10?") == "2025-01-10"


def test_route_after_rewrite_sends_rag_and_all_sources_to_orchestrator():
    """Both RAG-only and All-sources queries route through orchestrator; only internet_search goes to direct_tool_answer."""
    from rag_agent.query_rewriter import route_after_rewrite
    routes = route_after_rewrite({
        "questionIsClear": True,
        "rewrittenQuestions": [
            {"query": "q1", "forced_tool": None},
            {"query": "q2", "forced_tool": "internet_search"},
        ],
    })
    assert [r.node for r in routes] == ["orchestrator", "direct_tool_answer"]


def test_split_query_clauses_preserves_statutory_paragraphs_and_normalizes_breach_queries():
    """Statutory references like Article 52(1) and 52(2) are not split mid-citation and queries are not silently mutated."""
    from rag_agent.tools import _split_query_clauses
    q1 = "5/10/40 concentration limits Article 52(1) and 52(2) UCITS Directive 2009/65/EC Circular CSSF 24/856 active breach"
    assert _split_query_clauses(q1) == [q1]
    assert len(_split_query_clauses(q1)) == 1
    # Genuine numbered sub-questions (preceded by whitespace) still split cleanly
    numbered = "(1) Calculate the exact quarterly management fee for Class I (2) Calculate the subscription tax for Class R"
    assert len(_split_query_clauses(numbered)) == 3


def test_calculate_decimal_auto_sum_guards_mixed_units_and_explicit_totals():
    """calculate_decimal evaluates exact formulas and cross-line references without injecting unrequested [AUTO-SUM] lines."""
    from rag_agent.tools import calculate_decimal
    out_with_aggregate = calculate_decimal.invoke({
        "expressions": [
            "Alpha New EURM = 21.6 + 7.2",
            "Alpha New % = 28.8 / 240 * 100",
            "Beta % = 19.2 / 240 * 100",
            "Gamma % = 14.4 / 240 * 100",
            "Aggregate >5% = (Alpha New %) + (Beta %) + (Gamma %)",
        ]
    })
    assert "**26.00**" in out_with_aggregate
    assert "42.80" not in out_with_aggregate
    assert "[AUTO-SUM" not in out_with_aggregate

    out_homogeneous_legs = calculate_decimal.invoke({
        "expressions": [
            "Tier 1 = 50000000 * 0.004 * 91 / 365",
            "Tier 2 = 450000000 * 0.003 * 91 / 365",
            "Total Class I = (Tier 1) + (Tier 2)",
        ]
    })
    assert "[AUTO-SUM" not in out_homogeneous_legs
    assert "**386,438.35**" in out_homogeneous_legs


def test_multiturn_assistant_mode_preserves_history_context(monkeypatch):
    """Multi-turn follow-up queries in Assistant mode (and /all_sources) must preserve history_context and run QueryAnalysis."""
    from langchain_core.messages import HumanMessage
    import rag_agent.nodes_edges as ne_mod
    from rag_agent.state import QueryAnalysis, SubQuery

    captured_prompts = []

    class _DummyStructuredLLM:
        def invoke(self, msgs):
            captured_prompts.append(msgs[-1].content)
            return QueryAnalysis(
                is_clear=True,
                is_out_of_domain=False,
                questions=[SubQuery(query="What is the tolerance threshold for Bond UCIs under CSSF 24/856?", forced_tool=None)],
            )

    class _DummyLLM:
        def with_config(self, **_kwargs):
            return self

        def with_structured_output(self, _schema):
            return _DummyStructuredLLM()

    monkeypatch.setattr(ne_mod, "llm", _DummyLLM())

    state = {
        "source_mode": "Assistant mode",
        "conversation_summary": "",
        "messages": [
            HumanMessage(content="Under CSSF 24/856, what is the MMF threshold?", id="m1"),
            AIMessage(content="### CONCLUSION\n\n0.20% (20 bps).", id="m2"),
            HumanMessage(content="And what about Bond UCIs?", id="m3"),
        ],
    }
    out = ne_mod.rewrite_query(state)
    assert out["questionIsClear"] is True
    assert "Recent Conversation History:" in out["history_context"]
    assert "MMF threshold" in out["history_context"]
    assert out["rewrittenQuestions"][0]["forced_tool"] is None
    assert out["forced_tool"] is None
    assert len(captured_prompts) == 1
    assert "Recent Conversation History:" in captured_prompts[0]

    # Also verify /all_sources on Turn 2+ no longer short-circuits before building history_context
    state_slash = {
        "source_mode": "RAG only",
        "conversation_summary": "Prior discussion of CSSF 24/856 Table 1.",
        "messages": [
            HumanMessage(content="What is in Table 1?", id="s1"),
            AIMessage(content="MMF is 20 bps.", id="s2"),
            HumanMessage(content="/all_sources What about AIFMD leverage?", id="s3"),
        ],
    }
    out_slash = ne_mod.rewrite_query(state_slash)
    assert out_slash["questionIsClear"] is True
    assert "Prior discussion of CSSF 24/856 Table 1." in out_slash["history_context"]
    assert out_slash["rewrittenQuestions"][0]["forced_tool"] is None


def test_three_stage_model_resolution_and_backward_compatibility(monkeypatch):
    """Verify 3-stage model resolution and strict instance identity when all 3 stages use the same model."""
    import config as cfg_mod

    # 1. Backward compatibility when stage env vars are unset (or set to the same model)
    for stage in (1, 2, 3):
        monkeypatch.delenv(f"OPENAI_MODEL_REQUEST_{stage}", raising=False)
        monkeypatch.delenv(f"LLM_MODEL_REQUEST_{stage}", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    m1 = cfg_mod.resolve_model_name(stage=1, provider="openai")
    m2 = cfg_mod.resolve_model_name(stage=2, provider="openai")
    m3 = cfg_mod.resolve_model_name(stage=3, provider="openai")
    assert m1 == m2 == m3 == "gpt-4o"

    # Explicit identical setting across all 3 stages must resolve and cache to the exact same object
    monkeypatch.setenv("LLM_MODEL_REQUEST_1", "gpt-4o")
    monkeypatch.setenv("LLM_MODEL_REQUEST_2", "gpt-4o")
    monkeypatch.setenv("LLM_MODEL_REQUEST_3", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
    l1 = cfg_mod.create_llm(cfg_mod.resolve_model_name(1, "openai"), provider="openai")
    l2 = cfg_mod.create_llm(cfg_mod.resolve_model_name(2, "openai"), provider="openai")
    l3 = cfg_mod.create_llm(cfg_mod.resolve_model_name(3, "openai"), provider="openai")
    assert l1 is l2 is l3

    # 2. Tiered cost-saving configuration (Request 1 & 2 = gpt-4o-mini, Request 3 = gpt-4o)
    monkeypatch.setenv("LLM_MODEL_REQUEST_1", "gpt-4o-mini")
    monkeypatch.setenv("LLM_MODEL_REQUEST_2", "gpt-4o-mini")
    monkeypatch.setenv("LLM_MODEL_REQUEST_3", "gpt-4o")
    assert cfg_mod.resolve_model_name(1, "openai") == "gpt-4o-mini"
    assert cfg_mod.resolve_model_name(2, "openai") == "gpt-4o-mini"
    assert cfg_mod.resolve_model_name(3, "openai") == "gpt-4o"
    t1 = cfg_mod.create_llm(cfg_mod.resolve_model_name(1, "openai"), provider="openai")
    t2 = cfg_mod.create_llm(cfg_mod.resolve_model_name(2, "openai"), provider="openai")
    t3 = cfg_mod.create_llm(cfg_mod.resolve_model_name(3, "openai"), provider="openai")
    assert t1 is t2
    assert t1 is not t3

    # 3. Toggleable LLM response cache via configure_llm_cache
    from langchain_core.globals import get_llm_cache
    from langchain_core.messages import ToolMessage
    import rag_agent.nodes_edges as ne_mod
    cfg_mod.configure_llm_cache(True)
    assert get_llm_cache() is not None
    # Verify sanitize_agent_messages_for_llm strips volatile artifact/latency_ms so Request 2 & 3 hit cache
    tm1 = ToolMessage(content="chunk", tool_call_id="tc1", name="search_documents", artifact={"latency_ms": 5})
    tm2 = ToolMessage(content="chunk", tool_call_id="tc1", name="search_documents", artifact={"latency_ms": 19})
    assert ne_mod.sanitize_agent_messages_for_llm([tm1]) == ne_mod.sanitize_agent_messages_for_llm([tm2])
    cfg_mod.configure_llm_cache(False)
    assert get_llm_cache() is None


def test_orchestrator_routes_across_three_request_models(monkeypatch):
    """Verify orchestrator dispatches Request 1 (plan), Request 2 (calc gate), and Request 3 (synthesis) to their respective stage LLMs."""
    from langchain_core.messages import HumanMessage, ToolMessage
    import rag_agent.nodes_edges as ne_mod

    invoked_stages: list[str] = []

    class _StageMockLLM:
        def __init__(self, tag: str, response: AIMessage):
            self.tag = tag
            self._response = response

        def bind_tools(self, _tools):
            return self

        def invoke(self, _msgs, _config=None):
            invoked_stages.append(self.tag)
            return self._response

    req1_resp = AIMessage(
        content="[REASONING]\nSearching prospectus.",
        tool_calls=[{"name": "search_documents", "args": {"query": "Class I management fee"}, "id": "tc_1"}],
    )
    req2_resp = AIMessage(
        content="[REASONING]\nComputing fee.",
        tool_calls=[{"name": "calculate_decimal", "args": {"expressions": ["Fee = 50000000 * 0.004 * 91 / 365"]}, "id": "tc_2"}],
    )
    req3_resp = AIMessage(
        content="[REASONING]\nDone.\n[CONCLUSION]\n### CONCLUSION\nFee is 49,863.01 EUR.\n### DETAILED ANALYSIS\nVerified.\n---\n**SOURCES**\n- lux_alpha_ucits_prospectus.md",
    )

    monkeypatch.setattr(ne_mod, "llm_request_1", _StageMockLLM("gpt-4o-mini#req1", req1_resp))
    monkeypatch.setattr(ne_mod, "llm_request_2", _StageMockLLM("gpt-4o-mini#req2", req2_resp))
    monkeypatch.setattr(ne_mod, "llm_request_3", _StageMockLLM("gpt-4o#req3", req3_resp))

    q = "Calculate the exact management fee in EUR for 50,000,000 EUR over 91 days at 0.40%."
    # Turn 1 -> Request 1 (Retrieval Planning)
    out1 = ne_mod.orchestrator({"question": q, "forced_tool": "search_documents", "agent_messages": []}, {})
    msgs_after_t1 = out1["agent_messages"] + [
        ToolMessage(content="Document ID: lux_alpha\nContent: 0.40% Actual/365", tool_call_id="tc_1", name="search_documents")
    ]
    # Turn 2 -> Request 2 (Calculation Gate, since query is quantitative and no math tool has run yet)
    out2 = ne_mod.orchestrator({"question": q, "forced_tool": "search_documents", "agent_messages": msgs_after_t1}, {})
    msgs_after_t2 = msgs_after_t1 + out2["agent_messages"] + [
        ToolMessage(content="- Fee: `50000000 * 0.004 * 91 / 365` => **49,863.01**", tool_call_id="tc_2", name="calculate_decimal")
    ]
    # Turn 3 -> Request 3 (Final Audit Synthesis, since math tool has now executed)
    ne_mod.orchestrator({"question": q, "forced_tool": "search_documents", "agent_messages": msgs_after_t2}, {})

    assert invoked_stages == ["gpt-4o-mini#req1", "gpt-4o-mini#req2", "gpt-4o#req3"]


def test_openai_seed_configuration(monkeypatch):
    """Verify OPENAI_SEED is properly parsed and injected into ChatOpenAI instances."""
    import config as cfg_mod

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
    monkeypatch.setenv("OPENAI_SEED", "42")
    cfg_mod._LLM_INSTANCE_CACHE.clear()

    llm = cfg_mod.create_llm("gpt-4o-seed-test", provider="openai")
    assert getattr(llm, "seed", None) == 42

    monkeypatch.setenv("OPENAI_SEED", "invalid_seed")
    cfg_mod._LLM_INSTANCE_CACHE.clear()
    llm_invalid = cfg_mod.create_llm("gpt-4o-seed-test-invalid", provider="openai")
    assert getattr(llm_invalid, "seed", None) is None
