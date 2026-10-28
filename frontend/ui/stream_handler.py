"""Chat streaming: LangGraph thread management and the token-level Gradio generator.

Owns `thread_config` (conversation thread switching) so its definition and all
rebinding stay in one module.
"""

from __future__ import annotations

import logging
import re
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from config import RECURSION_LIMIT
from db import chat_store
from rag_agent.graph import agent_graph
from rag_agent.harness import sanitize_plain_text_math, split_reasoning_and_conclusion
from rag_agent.nodes_edges import content_to_text  # lane routing for streamed tool messages
from ui.renderers import (
    ERROR_NOTICE,
    FINAL_MARKERS,
    REASONING_MARKERS,
    STRINGS,
    clean_display_sources,
    close_open_divs,
    finalize_reasoning,
    format_rag_telemetry_html,
    get_display_html,
)

logger = logging.getLogger(__name__)


def create_thread_id():
    """Generate a unique thread ID for each conversation"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}


thread_config = create_thread_id()


def _seed_messages(conv_id: str) -> list[HumanMessage | AIMessage]:
    """Saved turns as LangChain messages, excluding the just-saved current user message."""
    msgs: list[HumanMessage | AIMessage] = []
    for role, plain in chat_store.plain_messages(conv_id)[:-1]:
        if role == "user":
            msgs.append(HumanMessage(content=plain))
        elif plain:
            msgs.append(AIMessage(content=plain))
    return msgs


def _chat_stream(message, history, conv_id, source_mode: str = "RAG only"):
    global thread_config
    effective_mode = "Assistant mode" if source_mode == "Assistant mode" else "RAG only"

    # Each conversation owns its LangGraph thread ("chat-<id>"). Switching
    # conversations (new chat or one restored from the sidebar) drops the
    # outgoing thread and seeds the new one with the saved turns, so context
    # survives restarts and never leaks across conversations.
    want_thread = f"chat-{conv_id}"
    if thread_config["configurable"]["thread_id"] != want_thread:
        try:
            agent_graph.checkpointer.delete_thread(thread_config["configurable"]["thread_id"])
        except AttributeError:
            pass
        thread_config = {
            "configurable": {"thread_id": want_thread, "recursion_limit": RECURSION_LIMIT}
        }

    current_state = agent_graph.get_state(thread_config)
    if current_state.next:
        agent_graph.update_state(
            thread_config,
            {"messages": [HumanMessage(content=message.strip())], "source_mode": effective_mode},
        )
        inputs = None
    else:
        if not current_state.values.get("messages"):
            prior = _seed_messages(conv_id)
            if prior:
                agent_graph.update_state(thread_config, {"messages": prior})
        inputs = {"messages": [HumanMessage(content=message.strip())], "source_mode": effective_mode}

    turn_start_perf = time.perf_counter()
    turn_start_ms = int(time.time() * 1000)
    reasoning_done_elapsed = None

    def _reasoning_elapsed(is_done: bool) -> float:
        nonlocal reasoning_done_elapsed
        if is_done:
            if reasoning_done_elapsed is None:
                reasoning_done_elapsed = max(0.1, time.perf_counter() - turn_start_perf)
            return reasoning_done_elapsed
        return max(0.1, time.perf_counter() - turn_start_perf)

    # Show the initial spinner
    yield get_display_html("", "", "waiting", STRINGS["status_analyzing"])

    reasoning_accumulated = ""
    final_accumulated = ""
    enforced_final_message = ""
    current_phase = "waiting"
    waiting_status = STRINGS["status_analyzing"]
    current_msg_id = None
    current_msg_phase = None
    current_msg_has_tools = False
    msg_buffer = ""
    reasoning_finalized = False

    def append_stream_text(text: str, phase: str):
        """Route text to the right lane: 'final' appends the answer, otherwise the reasoning box."""
        nonlocal reasoning_accumulated, final_accumulated, reasoning_finalized
        if not text:
            return
        if phase == "final":
            if reasoning_accumulated and not reasoning_finalized:
                reasoning_accumulated = finalize_reasoning(reasoning_accumulated)
                reasoning_finalized = True
            final_accumulated += text
        else:
            reasoning_accumulated += text

    # Enable stream_mode="messages" to receive token-level chunks
    for chunk, metadata in agent_graph.stream(inputs, thread_config, stream_mode="messages"):
        node = metadata.get("langgraph_node", "")

        # Tool events switch to a retrieval waiting spinner and record the action
        if getattr(chunk, "type", "") in ["tool", "ToolMessage"]:
            if msg_buffer:
                append_stream_text(msg_buffer, "final" if current_msg_phase == "final" else "reasoning")
                msg_buffer = ""
                current_msg_id = None
                current_msg_phase = None
                current_msg_has_tools = False

            artifact = getattr(chunk, "artifact", None)
            if isinstance(artifact, dict) and artifact.get("type") == "rag_telemetry":
                tool_pill = format_rag_telemetry_html(artifact)
            else:
                tool_name = getattr(chunk, "name", "tool") or "tool"
                raw_content = str(getattr(chunk, "content", ""))
                doc_matches = re.findall(r"(?:Document ID|File Name):\s*([A-Za-z0-9_.-]+)", raw_content)
                unique_docs = list(dict.fromkeys(doc_matches))

                if unique_docs:
                    tool_step_desc = f"📚 Retrieved statutory evidence from {', '.join(unique_docs)} via <code>{tool_name}</code>."
                elif tool_name == "calculate_decimal":
                    tool_step_desc = f"🔢 Computed symbolic arithmetic via <code>{tool_name}</code>."
                elif tool_name == "run_nav_audit":
                    tool_step_desc = f"⚖️ Executed NAV compliance audit via <code>{tool_name}</code>."
                elif tool_name == "internet_search":
                    tool_step_desc = f"🌐 Retrieved web reference via <code>{tool_name}</code>."
                else:
                    tool_step_desc = f"⚙️ Executed tool <code>{tool_name}</code>."
                tool_pill = f'<div class="reasoning-tool-pill">{tool_step_desc}</div>'

            if tool_pill not in reasoning_accumulated:
                if reasoning_accumulated and not reasoning_accumulated.endswith("\n"):
                    reasoning_accumulated += "\n"
                reasoning_accumulated += tool_pill + "\n"

            current_phase = "waiting"
            waiting_status = STRINGS["status_retrieving"]
            display_response = get_display_html(
                reasoning_accumulated,
                final_accumulated,
                current_phase,
                waiting_status,
                elapsed_s=_reasoning_elapsed(False),
                start_ms=turn_start_ms,
            )
            display_response = clean_display_sources(display_response)
            yield display_response
            continue

        # Only stream output from content-producing nodes
        if node in ["direct_answer", "orchestrator", "fallback_response", "aggregate_answers", "direct_tool_answer"]:
            if getattr(chunk, "type", "") in ["ai", "AIMessageChunk"]:
                has_tool_payload = bool(getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None))
                content_str = content_to_text(chunk.content)

                if current_msg_id != chunk.id:
                    if current_msg_id is not None and msg_buffer:
                        append_stream_text(msg_buffer, "final" if current_msg_phase == "final" else "reasoning")
                        msg_buffer = ""
                    current_msg_id = chunk.id
                    current_msg_phase = None
                    current_msg_has_tools = has_tool_payload
                elif has_tool_payload:
                    current_msg_has_tools = True

                if content_str:
                    is_chunk = type(chunk).__name__ == "AIMessageChunk"
                    # Full (non-chunk) messages are only shown while nothing has
                    # been streamed yet; afterwards they duplicate visible content.
                    if not is_chunk and node not in ["direct_answer"] and (final_accumulated or msg_buffer):
                        if node == "aggregate_answers" and content_str:
                            enforced_final_message = content_str
                        continue

                    # Classify output and detect reasoning/conclusion markers
                    if node == "orchestrator":
                        msg_buffer += content_str
                        content_str = ""

                        # Tool-planning turns stay locked in 'reasoning'
                        if current_msg_has_tools:
                            for marker in REASONING_MARKERS:
                                idx = msg_buffer.upper().find(marker)
                                if idx != -1:
                                    msg_buffer = msg_buffer[:idx] + msg_buffer[idx + len(marker):]
                                    break
                            current_msg_phase = "reasoning"
                        elif current_msg_phase != "final":
                            found_final = False
                            for marker in FINAL_MARKERS:
                                idx = msg_buffer.upper().find(marker)
                                if idx != -1:
                                    before_marker = msg_buffer[:idx]
                                    if before_marker:
                                        reasoning_accumulated += before_marker
                                    current_msg_phase = "final"
                                    msg_buffer = msg_buffer[idx + len(marker):]
                                    found_final = True
                                    break

                            if not found_final and current_msg_phase is None:
                                for marker in REASONING_MARKERS:
                                    idx = msg_buffer.upper().find(marker)
                                    if idx != -1:
                                        current_msg_phase = "reasoning"
                                        msg_buffer = msg_buffer[:idx] + msg_buffer[idx + len(marker):]
                                        break

                            # Hard Barrier 2.4 — Transition to 'final' if structural conclusion header arrives without [CONCLUSION]
                            if not found_final:
                                m_anchor = re.search(
                                    r"(?:^|\n)\s*(###\s*CONCLUSION\b|⚠️\s*\*?\*?Not found:)",
                                    msg_buffer,
                                    flags=re.IGNORECASE,
                                )
                                if m_anchor:
                                    anchor_start = m_anchor.start(1)
                                    before_anchor = msg_buffer[:anchor_start]
                                    if before_anchor:
                                        reasoning_accumulated += before_anchor
                                    current_msg_phase = "final"
                                    msg_buffer = msg_buffer[anchor_start:]
                                    found_final = True
                        # Hold back the last 30 chars in case a marker is split across chunks
                        if len(msg_buffer) > 30:
                            content_str = msg_buffer[:-30]
                            msg_buffer = msg_buffer[-30:]
                        node_phase = current_msg_phase if current_msg_phase else "reasoning"
                    else:
                        if any(m.upper() in content_str.upper() for m in (*REASONING_MARKERS, *FINAL_MARKERS)):
                            r_part, content_str = split_reasoning_and_conclusion(content_str)
                            if r_part and not reasoning_accumulated:
                                reasoning_accumulated += r_part
                        node_phase = "final"
                    current_phase = node_phase
                    append_stream_text(content_str, node_phase)
                    # Yield the current display state
                    display_reasoning = reasoning_accumulated + msg_buffer if current_phase == "reasoning" else reasoning_accumulated
                    display_final = final_accumulated + msg_buffer if current_phase == "final" else final_accumulated
                    is_done_now = (current_phase == "final") or bool(display_final and display_final.strip())
                    display_response = get_display_html(
                        display_reasoning,
                        display_final,
                        current_phase,
                        waiting_text=STRINGS["status_thinking"] if current_phase == "reasoning" else STRINGS["status_answering"],
                        elapsed_s=_reasoning_elapsed(is_done_now),
                        start_ms=turn_start_ms,
                    )
                    display_response = clean_display_sources(display_response)
                    yield display_response

    # Flush whatever is left in the buffer when the stream ends
    append_stream_text(msg_buffer, current_phase)

    # Strip any residual [REASONING] / [CONCLUSION] markers that might remain in final_accumulated
    if final_accumulated and any(m.upper() in final_accumulated.upper() for m in (*REASONING_MARKERS, *FINAL_MARKERS)):
        r_extra, final_accumulated = split_reasoning_and_conclusion(final_accumulated)
        if r_extra and not reasoning_accumulated:
            reasoning_accumulated = r_extra

    if enforced_final_message:
        r_enf, c_enf = split_reasoning_and_conclusion(enforced_final_message)
        if c_enf:
            final_accumulated = c_enf
        if r_enf and not reasoning_accumulated:
            reasoning_accumulated = r_enf

    # Safe fallback: if final_accumulated is still empty, pull the answer from graph state
    if not final_accumulated:
        try:
            latest_state = agent_graph.get_state(thread_config)
            msgs = latest_state.values.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content:
                    final_text = content_to_text(m.content)
                    if final_text:
                        r_part, c_part = split_reasoning_and_conclusion(final_text)
                        if r_part and not reasoning_accumulated:
                            reasoning_accumulated = r_part
                        final_accumulated = c_part or final_text
                        break
            if not final_accumulated:
                ans_list = latest_state.values.get("agent_answers", [])
                if ans_list and ans_list[0].get("answer"):
                    r_part, c_part = split_reasoning_and_conclusion(ans_list[0]["answer"])
                    if r_part and not reasoning_accumulated:
                        reasoning_accumulated = r_part
                    final_accumulated = c_part or ans_list[0]["answer"]
        except Exception as e:
            logger.warning("Fallback message extraction failed: %s", e)

    # Final check to append the conclusion line if never appended
    if reasoning_accumulated and not reasoning_finalized:
        reasoning_accumulated = finalize_reasoning(reasoning_accumulated)
        reasoning_finalized = True

    # Close any open div tag in reasoning_accumulated at the very end
    reasoning_accumulated = close_open_divs(reasoning_accumulated)
    if final_accumulated:
        final_accumulated = sanitize_plain_text_math(final_accumulated)

    display_response = get_display_html(
        reasoning_accumulated,
        final_accumulated,
        "final",
        is_completed=True,
        elapsed_s=_reasoning_elapsed(True),
        start_ms=turn_start_ms,
    )
    display_response = clean_display_sources(display_response)
    # Never persist an empty turn: the UI uses "" as the "pending reply" marker
    if not display_response:
        display_response = ERROR_NOTICE

    try:
        chat_store.add_message(conv_id, "assistant", display_response, plain=final_accumulated)
    except Exception as e:
        logger.warning("Failed to persist assistant turn: %s", e)

    yield display_response
