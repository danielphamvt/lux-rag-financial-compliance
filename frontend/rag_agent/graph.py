from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from rag_agent.context_compactor import compress_context, should_compress_context
from rag_agent.nodes_edges import (
    aggregate_answers,
    collect_answer,
    direct_tool_answer,
    fallback_response,
    is_greeting,
    orchestrator,
    route_after_orchestrator_call,
)
from rag_agent.query_rewriter import (
    direct_answer,
    request_clarification,
    rewrite_query,
    route_after_rewrite,
    summarize_history,
)
from rag_agent.state import State, ToolExecutionRecord
from rag_agent.tools import calculate_decimal, run_nav_audit, search_documents
from skills.internet_search.skill import internet_search

checkpointer = InMemorySaver()

# --- BUILD FLATTENED MAIN GRAPH ---
graph_builder = StateGraph(State)

# Add standard nodes
graph_builder.add_node(direct_answer)
graph_builder.add_node(summarize_history)
graph_builder.add_node(rewrite_query)
graph_builder.add_node(request_clarification)
graph_builder.add_node(direct_tool_answer)
graph_builder.add_node(aggregate_answers)

# --- Ledger-backed tool node ---
# The prebuilt ToolNode executes tool calls and returns ToolMessages; this thin
# wrapper also appends one ToolExecutionRecord per real execution so the durable
# ledger (not the compressible message stream) stays the audit evidence source.
_TOOL_NODE = ToolNode(
    [search_documents, run_nav_audit, calculate_decimal, internet_search],
    messages_key="agent_messages",
)


def agent_tools(state: State) -> dict:
    """Execute pending tool calls and record every real execution in the ledger.

    Args:
        state: Current graph state carrying `agent_messages` with the pending AIMessage tool calls.

    Returns:
        ToolMessages for `agent_messages` plus one `ToolExecutionRecord` per execution.
    """
    out = _TOOL_NODE.invoke(state)
    pending_args: dict[str, dict] = {}
    for m in state.get("agent_messages", []):
        if isinstance(m, AIMessage):
            for tc in getattr(m, "tool_calls", None) or []:
                pending_args[tc.get("id", "")] = tc.get("args", {}) or {}
    records = [
        ToolExecutionRecord(
            tool_call_id=getattr(tm, "tool_call_id", ""),
            tool_name=getattr(tm, "name", "") or "",
            args=pending_args.get(getattr(tm, "tool_call_id", ""), {}),
            output=str(getattr(tm, "content", "")),
        )
        for tm in out.get("agent_messages", [])
        if getattr(tm, "type", "") == "tool"
    ]
    out["tool_execution_log"] = records
    return out


# Add agent nodes directly to the main graph
graph_builder.add_node("orchestrator", orchestrator)
graph_builder.add_node("agent_tools", agent_tools)
graph_builder.add_node("compress_context", compress_context)
graph_builder.add_node("fallback_response", fallback_response)
graph_builder.add_node("should_compress_context", should_compress_context)
graph_builder.add_node("collect_answer", collect_answer)

# Routing and lifecycle edges
graph_builder.add_conditional_edges(START, is_greeting)
graph_builder.add_edge("direct_answer", END)
graph_builder.add_edge("summarize_history", "rewrite_query")
graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
graph_builder.add_edge("request_clarification", "rewrite_query")

# Agent loop edges
graph_builder.add_conditional_edges(
    "orchestrator", 
    route_after_orchestrator_call, 
    {"tools": "agent_tools", "fallback_response": "fallback_response", "collect_answer": "collect_answer"}
)
graph_builder.add_edge("agent_tools", "should_compress_context")
graph_builder.add_edge("compress_context", "orchestrator")
graph_builder.add_edge("fallback_response", "collect_answer")

# Parallel joins to aggregate_answers
graph_builder.add_edge("collect_answer", "aggregate_answers")
graph_builder.add_edge("direct_tool_answer", "aggregate_answers")
graph_builder.add_edge("aggregate_answers", END)

# Compile with memory and interrupt on clarification
agent_graph = graph_builder.compile(
    checkpointer=checkpointer, 
    interrupt_before=["request_clarification"]
)
