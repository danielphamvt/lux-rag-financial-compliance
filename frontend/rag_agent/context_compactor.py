"""Context compaction helpers: token estimation, history formatting, and the context-compression graph nodes."""

from typing import Any, Literal

import tiktoken
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from config import COMPRESS_TOKEN_THRESHOLD, TOKEN_GROWTH_FACTOR
from rag_agent.harness import build_compressed_record_messages as _build_compressed_record_messages
from rag_agent.nodes_edges import content_to_text, get_request_llm
from rag_agent.prompts import get_context_compression_prompt
from rag_agent.state import State

try:
    _ENCODING = tiktoken.get_encoding("o200k_base")
except Exception:
    try:
        _ENCODING = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        _ENCODING = tiktoken.get_encoding("cl100k_base")


def estimate_context_tokens(messages: list) -> int:
    """Estimate the total token count across a sequence of LangChain messages.

    Args:
        messages: Sequence of message objects with `.content` attributes.

    Returns:
        Total token count estimated via `tiktoken`.
    """
    return sum(len(_ENCODING.encode(str(msg.content))) for msg in messages if hasattr(msg, "content") and msg.content)


def format_history_lines(messages: list, keep: int = 6) -> str:
    """Join recent conversational messages as `Role: content` lines, skipping tool calls.

    Args:
        messages: Sequence of conversation messages.
        keep: Maximum number of recent non-tool messages to include.

    Returns:
        Newline-joined string of `User:` and `Assistant:` turns.
    """
    relevant = [
        m for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
    ]
    return "".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {content_to_text(m.content)}\n"
        for m in relevant[-keep:]
    )


def should_compress_context(state: State) -> Command[Literal["compress_context", "orchestrator"]]:
    """Route to `compress_context` when accumulated message tokens exceed the dynamic threshold.

    Args:
        state: Current LangGraph `State`.

    Returns:
        LangGraph `Command` updating `retrieval_keys` and routing to `compress_context` or `orchestrator`.
    """
    messages = state["agent_messages"]

    new_ids: set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "search_documents":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = COMPRESS_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(update={"retrieval_keys": updated_ids}, goto=goto)


def compress_context(state: State) -> dict[str, Any]:
    """Summarize prior `agent_messages` into `context_summary` and prune compressed messages.

    Args:
        state: Current LangGraph `State`.

    Returns:
        State update dict with `context_summary` and `RemoveMessage` entries for `agent_messages`.
    """
    messages = state["agent_messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    to_remove = messages[1:]
    for msg in to_remove:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = get_request_llm(1).invoke([SystemMessage(content=get_context_compression_prompt()), HumanMessage(content=conversation_text)])
    new_summary = summary_response.content

    retrieved_ids: set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))
        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {
        "context_summary": new_summary,
        "agent_messages": [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]
        + _build_compressed_record_messages(state.get("tool_execution_log") or []),
    }
