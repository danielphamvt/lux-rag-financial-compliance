"""Query-rewriting graph nodes: small-talk answers, history summarization, query rewrite, and rewrite routing."""

from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.types import Send

from config import MAX_HISTORY_MESSAGES, MAX_HISTORY_TOKENS
from domain_profile import PROFILE, get_prompt
from rag_agent.context_compactor import estimate_context_tokens, format_history_lines
from rag_agent.nodes_edges import _turn_reset, content_to_text, get_request_llm
from rag_agent.prompts import get_conversation_summary_prompt, get_rewrite_query_prompt
from rag_agent.routing import (
    _DOMAIN_SIGNAL_RE,
    is_out_of_domain_message,
    normalize_message,
)
from rag_agent.state import QueryAnalysis, State

# --- DIRECT ANSWER ROUTER ---
def direct_answer(state: State):
    raw_content = state["messages"][-1].content
    if is_out_of_domain_message(raw_content):
        return {"messages": [AIMessage(content=PROFILE["strings"]["out_of_scope_notice"])]}
    msg = normalize_message(raw_content)
    small_talk = PROFILE["small_talk"]
    if any(t in msg for t in small_talk["thanks_triggers"]):
        ans = small_talk["thanks_response"]
    elif any(t in msg for t in small_talk["farewell_triggers"]):
        ans = small_talk["farewell_response"]
    else:
        ans = small_talk["default_response"]
    return {"messages": [AIMessage(content=ans)]}

# --- MAIN GRAPH NODES & EDGES ---
def summarize_history(state: State):
    messages = state["messages"]
    
    if len(messages) <= MAX_HISTORY_MESSAGES and estimate_context_tokens(messages) <= MAX_HISTORY_TOKENS:
        return {"conversation_summary": ""}

    history_lines = format_history_lines(messages[:-1])
    if not history_lines:
        return {"conversation_summary": ""}

    conversation = "Conversation history:\n" + history_lines

    summary_response = get_request_llm(1).with_config(temperature=0.2).invoke([SystemMessage(content=get_conversation_summary_prompt()), HumanMessage(content=conversation)])
    return {"conversation_summary": content_to_text(summary_response.content), "agent_answers": [{"__reset__": True}]}

def rewrite_query(state: State):
    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

    query_text = last_message.content.strip()
    source_mode = state.get("source_mode", "RAG only")
    # Default for high-precision financial compliance: RAG only ("search_documents"); Assistant mode sets None
    default_forced_tool = None if source_mode == "Assistant mode" else "search_documents"
    explicit_cmd = False
    if query_text.startswith("/internet_search"):
        parts = query_text.split(" ", 1)
        query_text = parts[1].strip() if len(parts) > 1 else ""
        default_forced_tool = "internet_search"
        explicit_cmd = True
    elif query_text.startswith("/all_sources"):
        parts = query_text.split(" ", 1)
        query_text = parts[1].strip() if len(parts) > 1 else ""
        default_forced_tool = None
        explicit_cmd = True
    elif query_text.startswith("/search_documents"):
        parts = query_text.split(" ", 1)
        query_text = parts[1].strip() if len(parts) > 1 else ""
        default_forced_tool = "search_documents"
        explicit_cmd = True

    if explicit_cmd and not query_text:
        return {"questionIsClear": False, "messages": [AIMessage(content=PROFILE["strings"]["forced_tool_empty_query"])]}

    if is_out_of_domain_message(query_text):
        return {"questionIsClear": False, "messages": [AIMessage(content=PROFILE["strings"]["out_of_scope_notice"])]}

    # Optimization 1: Skip rewrite if it's the first question
    if not conversation_summary.strip() and len(state["messages"]) <= 1:
        # Rule-based router for the first question to save an LLM call entirely.
        return {
            "questionIsClear": True,
            "messages": [],
            "originalQuery": query_text,
            "rewrittenQuestions": [{"query": query_text, "forced_tool": default_forced_tool}],
            "forced_tool": default_forced_tool,
            **_turn_reset(state),
        }

    recent_history_text = ""
    if not conversation_summary.strip() and len(state["messages"]) > 1:
        # History is too short to be summarized, pass raw recent messages
        history_lines = format_history_lines(state["messages"][:-1])
        if history_lines:
            recent_history_text = "Recent Conversation History:\n" + history_lines + "\n"

    history_context_out = (f"Conversation Summary:\n{conversation_summary}\n\n" if conversation_summary.strip() else "") + recent_history_text
    context_section = history_context_out + f"User Query:\n{query_text}\n"

    llm_with_structure = get_request_llm(1).with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    
    try:
        response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("LLM structured output validation failed: %s", e)
        # Fallback gracefully if JSON parsing fails
        clarification_msg = get_prompt("clarification_fallback")
        return {"questionIsClear": False, "messages": [AIMessage(content=clarification_msg)], "history_context": history_context_out}

    if getattr(response, "is_out_of_domain", False) and not _DOMAIN_SIGNAL_RE.search(query_text):
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=PROFILE["strings"]["out_of_scope_notice"])],
            "history_context": history_context_out,
        }

    if response and getattr(response, "questions", None) and getattr(response, "is_clear", False):
        # Keep the last MAX_HISTORY_MESSAGES messages as context
        messages_to_keep = state["messages"][-MAX_HISTORY_MESSAGES:]
        delete_old = [RemoveMessage(id=m.id) for m in state["messages"] if m not in messages_to_keep and not isinstance(m, SystemMessage)]
        orig_q = query_text
        combined_q = " ".join(q.query.strip() for q in response.questions if q.query.strip()) or orig_q
        full_q = f"{combined_q}\n\n[Original User Context: {orig_q}]" if (orig_q and orig_q != combined_q) else (combined_q or orig_q)
        if default_forced_tool == "internet_search":
            ft = "internet_search"
        elif default_forced_tool is None:
            ft = None
        else:
            ft = response.questions[0].forced_tool or default_forced_tool
        rewritten = [{"query": full_q, "forced_tool": ft}]
        return {
            "questionIsClear": True,
            "messages": delete_old,
            "originalQuery": query_text,
            "rewrittenQuestions": rewritten,
            "forced_tool": ft,
            "history_context": history_context_out,
            **_turn_reset(state),
        }

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else get_prompt("clarification_fallback")
    return {"questionIsClear": False, "messages": [AIMessage(content=clarification)], "history_context": history_context_out}

def request_clarification(state: State) -> dict:
    """No-op terminal node when clarification is requested from the user."""
    return {}


def route_after_rewrite(state: State) -> Literal["request_clarification"] | list[Send]:
    """Route to `request_clarification` if unclear, otherwise fan out `Send` commands to sub-agent nodes."""
    if not state.get("questionIsClear", False):
        return "request_clarification"
    routes: list[Send] = []
    hist_ctx = state.get("history_context", "")
    for idx, subquery in enumerate(state["rewrittenQuestions"]):
        q_text = subquery.get("query", "")
        ft = subquery.get("forced_tool")
        target = "direct_tool_answer" if ft == "internet_search" else "orchestrator"
        routes.append(Send(target, {"question": q_text, "question_index": idx, "forced_tool": ft, "history_context": hist_ctx}))
    return routes
