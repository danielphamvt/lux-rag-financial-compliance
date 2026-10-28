from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from config import MAX_ITERATIONS, MAX_TOOL_CALLS, llm, llm_request_1, llm_request_2, llm_request_3
from domain_profile import PROFILE
from rag_agent.harness import (
    build_compressed_record_messages as _build_compressed_record_messages,
    enforce_response_protocol,
    has_symbolic_math_tool_call as _has_symbolic_math_tool_call,
    is_quantitative_query as _is_quantitative_query,
    merge_prior_calc_legs as _merge_prior_calc_legs,
    sanitize_plain_text_math,
)
from rag_agent.prompts import (
    get_fallback_response_prompt,
    get_internet_no_results_prompt,
    get_internet_search_prompt,
    get_orchestrator_calculation_gate_prompt,
    get_orchestrator_plan_prompt,
    get_orchestrator_prompt,
    get_orchestrator_synthesis_prompt,
)
from rag_agent.routing import is_greeting_message, is_out_of_domain_message
from rag_agent.state import State, ToolExecutionRecord
from rag_agent.tools import calculate_decimal, run_nav_audit, search_documents
from skills.internet_search.skill import internet_search

_DEFAULT_LLM = llm
llm_with_tools = llm.bind_tools([search_documents, run_nav_audit, calculate_decimal, internet_search])


def get_request_llm(stage: Literal[1, 2, 3] = 3) -> Any:
    """Return the chat model configured for the given request stage (1, 2, or 3).

    Args:
        stage: Request tier (1 = keyword extraction & retrieval plan,
            2 = Python formula construction gate, 3 = final audit synthesis).

    Returns:
        Configured LangChain chat model for `stage`. When all 3 stages use the
        same model (or when `nodes_edges.llm` is monkeypatched in tests),
        returns the shared `llm` instance for 100% backward compatibility.
    """
    if llm is not _DEFAULT_LLM:
        return llm
    return llm_request_1 if stage == 1 else (llm_request_2 if stage == 2 else llm_request_3)


def _bind_orchestrator_tools(active_llm: Any, forced_tool: str | None) -> Any:
    """Bind the appropriate compliance/search toolset to `active_llm`."""
    if forced_tool == "internet_search":
        return active_llm.bind_tools([internet_search])
    if forced_tool == "search_documents":
        return active_llm.bind_tools([search_documents, run_nav_audit, calculate_decimal])
    if active_llm is llm:
        return llm_with_tools
    return active_llm.bind_tools([search_documents, run_nav_audit, calculate_decimal, internet_search])


def _turn_reset(state: State) -> dict[str, Any]:
    """Per-turn resets shared by every rewrite_query exit path that starts research."""
    return {
        "agent_answers": [{"__reset__": True}],
        "agent_messages": [RemoveMessage(id=m.id) for m in state.get("agent_messages", [])],
        "tool_call_count": {"__reset__": True},
        "iteration_count": {"__reset__": True},
        "retrieval_keys": {"__reset__": True},
        "context_summary": "",
    }


def is_greeting(state: State) -> Literal["direct_answer", "summarize_history"]:
    """Route greetings and out-of-domain messages to `direct_answer`, else `summarize_history`.

    Args:
        state: Current LangGraph `State`.

    Returns:
        Target node name (`"direct_answer"` or `"summarize_history"`).
    """
    last_content = state["messages"][-1].content
    if is_greeting_message(last_content) or is_out_of_domain_message(last_content):
        return "direct_answer"
    return "summarize_history"


def aggregate_answers(state: State) -> dict[str, list[AIMessage]]:
    """Aggregate completed sub-agent answers into the user-visible `messages` channel.

    Args:
        state: Current LangGraph `State`.

    Returns:
        State update dict containing the final `AIMessage`.
    """
    answers = state.get("agent_answers")
    if not answers:
        return {"messages": [AIMessage(content=PROFILE["strings"]["no_answers_generated"])]}
    return {"messages": [AIMessage(content=answers[0].get("answer", ""))]}


def direct_tool_answer(state: State, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Execute a forced `internet_search` tool call directly and synthesize a protocol-compliant answer.

    Args:
        state: Current LangGraph `State`.
        config: Optional LangChain `RunnableConfig`.

    Returns:
        State update dict with `agent_answers` and `tool_execution_log`.
    """
    search_query = state["question"]
    hist_ctx = state.get("history_context", "")
    search_result = internet_search.invoke({"query": search_query})

    no_results_msg = PROFILE["strings"]["internet_no_results"]
    has_results = search_result and "[INTERNET_DATA_START]" in search_result and no_results_msg not in search_result
    if not has_results:
        return {
            "agent_answers": [
                {"index": state["question_index"], "question": search_query, "answer": get_internet_no_results_prompt(search_query)}
            ]
        }

    prompt = get_internet_search_prompt(search_result, search_query, history_context=hist_ctx)
    response = get_request_llm(3).invoke([HumanMessage(content=prompt)], config)
    internet_record = ToolExecutionRecord(
        tool_name="internet_search",
        tool_call_id="direct_internet_search",
        args={"query": search_query},
        output=str(search_result or ""),
        is_recent=True,
    )
    raw_text = enforce_response_protocol(content_to_text(response.content), [], ledger=[internet_record])
    return {
        "agent_answers": [{"index": state["question_index"], "question": search_query, "answer": raw_text}],
        "tool_execution_log": [internet_record],
    }


def sanitize_agent_messages_for_llm(messages: list) -> list:
    """Ensure conversation history complies with OpenAI / Anthropic strict tool calling schema.

    Args:
        messages: Raw sequence of `agent_messages` from graph state.

    Returns:
        Sanitized list of LangChain messages with matched `ToolMessage` responses.
    """
    sanitized = []
    pending_tool_call_ids = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage) and pending_tool_call_ids:
            for missing_id in list(pending_tool_call_ids):
                sanitized.append(ToolMessage(content="No additional data retrieved.", tool_call_id=missing_id, name="tool"))
            pending_tool_call_ids.clear()

        if isinstance(msg, ToolMessage) and getattr(msg, "artifact", None) is not None:
            msg = msg.model_copy(update={"artifact": None, "response_metadata": {}})
        elif isinstance(msg, AIMessage):
            msg = msg.model_copy(update={"response_metadata": {}, "usage_metadata": None})
        sanitized.append(msg)

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc_id := tc.get("id"):
                    pending_tool_call_ids.add(tc_id)
        elif isinstance(msg, ToolMessage):
            pending_tool_call_ids.discard(getattr(msg, "tool_call_id", None))

    for missing_id in list(pending_tool_call_ids):
        sanitized.append(ToolMessage(content="No additional data retrieved.", tool_call_id=missing_id, name="tool"))
    return sanitized


def orchestrator(state: State, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Run the 3-stage ReAct orchestrator (Retrieval Plan -> Symbolic Calculation Gate -> Audit Synthesis).

    Args:
        state: Current LangGraph `State`.
        config: Optional LangChain `RunnableConfig`.

    Returns:
        State update dict advancing `agent_messages`, `tool_call_count`, and `iteration_count`.
    """
    forced_tool = state.get("forced_tool")
    rag_only = forced_tool == "search_documents"
    question = state.get("question", "")
    q_idx = state.get("question_index", 0)
    context_summary = state.get("context_summary", "").strip()
    hist_ctx = state.get("history_context", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt(rag_only=rag_only))

    combined_ctx = ""
    if context_summary:
        combined_ctx += f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}\n\n"
    if hist_ctx:
        combined_ctx += f"[CONVERSATION HISTORY]\n\n{hist_ctx}\n\n"

    summary_injection = [HumanMessage(content=combined_ctx.strip())] if combined_ctx else []
    base_state_out = {"question": question, "question_index": q_idx, "history_context": hist_ctx, "forced_tool": forced_tool}

    if not state.get("agent_messages"):
        bound_llm = _bind_orchestrator_tools(get_request_llm(1), forced_tool)
        human_msg = HumanMessage(content=question)
        plan_directive = HumanMessage(content=get_orchestrator_plan_prompt(forced_tool=forced_tool))
        response = bound_llm.invoke([sys_msg] + summary_injection + [human_msg, plan_directive], config)
        return {**base_state_out, "agent_messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    is_quant = _is_quantitative_query(question)
    ledger = state.get("tool_execution_log")
    has_math_call = _has_symbolic_math_tool_call(state["agent_messages"], ledger=ledger)
    if forced_tool != "internet_search" and is_quant and not has_math_call:
        active_stage: Literal[2, 3] = 2
        synth_text = get_orchestrator_calculation_gate_prompt(query_context=question)
    else:
        active_stage = 3
        has_calc_dec = any(
            isinstance(m, AIMessage) and any(tc.get("name") == "calculate_decimal" for tc in (getattr(m, "tool_calls", None) or []))
            for m in state["agent_messages"]
        ) or any(getattr(r, "tool_name", "") == "calculate_decimal" for r in (ledger or []))
        synth_text = get_orchestrator_synthesis_prompt(
            rag_only=rag_only,
            query_context=question,
            include_calc_gate=(is_quant and not has_calc_dec),
        )

    bound_llm = _bind_orchestrator_tools(get_request_llm(active_stage), forced_tool)
    synthesis_directive = HumanMessage(content=synth_text)
    sanitized_history = sanitize_agent_messages_for_llm(state["agent_messages"])
    response = bound_llm.invoke([sys_msg] + summary_injection + sanitized_history + [synthesis_directive], config)
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    if tool_calls:
        _merge_prior_calc_legs(state["agent_messages"], tool_calls, ledger)
    return {**base_state_out, "agent_messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}


def route_after_orchestrator_call(state: State) -> Literal["tools", "fallback_response", "collect_answer"]:
    """Route after `orchestrator` to `collect_answer`, `fallback_response`, or `tools`.

    Args:
        state: Current LangGraph `State`.

    Returns:
        Next graph node key (`"tools"`, `"fallback_response"`, or `"collect_answer"`).
    """
    last_message = state["agent_messages"][-1]
    if not (getattr(last_message, "tool_calls", None) or []):
        return "collect_answer"
    if state.get("iteration_count", 0) >= MAX_ITERATIONS or state.get("tool_call_count", 0) > MAX_TOOL_CALLS:
        return "fallback_response"
    return "tools"


def fallback_response(state: State, config: RunnableConfig | None = None) -> dict[str, list[AIMessage]]:
    """Synthesize a best-effort answer from accumulated tool outputs when iteration limits are reached.

    Args:
        state: Current LangGraph `State`.
        config: Optional LangChain `RunnableConfig`.

    Returns:
        State update dict appending the synthesized `AIMessage` to `agent_messages`.
    """
    rag_only = state.get("forced_tool") == "search_documents"
    seen: set[str] = set()
    unique_contents: list[str] = []
    for m in state["agent_messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()
    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n"
            + "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."
    agent_msgs = state.get("agent_messages", [])
    actual_query = (content_to_text(agent_msgs[0].content) if agent_msgs else "") or state.get("question") or state.get("originalQuery") or PROFILE["strings"]["no_query_fallback"]
    prompt_content = (
        f"USER QUERY: {actual_query}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nPlease answer the user query based on the system prompt rules."
    )
    response = get_request_llm(3).invoke([SystemMessage(content=get_fallback_response_prompt(rag_only=rag_only)), HumanMessage(content=prompt_content)], config)
    return {"agent_messages": [response]}


def content_to_text(content: Any) -> str:
    """Coerce message content (plain str or provider block list) to plain text.

    Args:
        content: Raw `AIMessage` or `HumanMessage` content payload.

    Returns:
        Sanitized plain-text string.
    """
    if isinstance(content, str):
        return sanitize_plain_text_math(content)
    if isinstance(content, list):
        raw = "".join(
            b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else (b if isinstance(b, str) else "")
            for b in content
        )
        return sanitize_plain_text_math(raw)
    return sanitize_plain_text_math(str(content)) if content else ""


def collect_answer(state: State) -> dict[str, list[dict[str, Any]]]:
    """Enforce the 5-marker response protocol on the final `AIMessage` and publish to `agent_answers`.

    Args:
        state: Current LangGraph `State`.

    Returns:
        State update dict appending the finalized answer entry to `agent_answers`.
    """
    agent_messages = state.get("agent_messages", [])
    ledger = state.get("tool_execution_log")
    last_message = agent_messages[-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls

    answer = PROFILE["strings"]["unable_to_answer"]
    if is_valid:
        raw_text = content_to_text(last_message.content)
        answer = enforce_response_protocol(raw_text, agent_messages, ledger=ledger)

    return {
        "agent_answers": [{"index": state["question_index"], "question": state["question"], "answer": answer}]
    }


def __getattr__(name: str) -> Any:
    """Lazily re-export `rewrite_query` from `rag_agent.query_rewriter` to avoid a circular import."""
    if name == "rewrite_query":
        from rag_agent import query_rewriter
        return query_rewriter.rewrite_query
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
