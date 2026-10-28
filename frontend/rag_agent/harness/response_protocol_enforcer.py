"""Streaming protocol enforcer & UI lane splitter (`response_protocol_enforcer.py`).

Enforces all 5 `[REASONING]` / `[CONCLUSION]` protocol markers, normalizes section
headings across backend and UI streams, and splits raw text into separate reasoning
and conclusion lanes.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

from domain_profile import PROFILE
from rag_agent.harness.citation_and_warning_guard import (
    extract_fallback_sources_footer,
    is_closed_abstention,
    normalize_external_source_warning,
    verify_sources_footer,
)
from rag_agent.harness.latex_to_unicode_normalizer import (
    sanitize_plain_text_math,
)


def normalize_protocol_headings(text: str) -> str:
    """Normalize Markdown section headings and source footers to uppercase protocol form.

    Args:
        text: Raw or partially streamed response text.

    Returns:
        Text with canonical `### CONCLUSION`, `### DETAILED ANALYSIS`, `**SOURCES**`,
        `**INTERNET SOURCES**`, and `### REFERENCE FROM THE INTERNET` headings.
    """
    if not text:
        return text
    text = re.sub(r"(?m)^#{1,4}[ \t]*Conclusion:?[ \t]*$", "### CONCLUSION", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^#{1,4}[ \t]*Detailed\s+Analysis:?[ \t]*$", "### DETAILED ANALYSIS", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\*\*Sources:?\*\*", "**SOURCES**", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^\*\*Internet\s+sources:?\*\*", "**INTERNET SOURCES**", text, flags=re.IGNORECASE)
    return re.sub(
        r"(?m)^#{1,4}[ \t]*Reference\s+from\s+the\s+Internet:?[ \t]*$",
        "### REFERENCE FROM THE INTERNET",
        text,
        flags=re.IGNORECASE,
    )


def _build_deterministic_reasoning(agent_messages: list | None = None, ledger: list | None = None) -> str:
    """Synthesize a concise reasoning scratchpad from prior `AIMessage` reasoning or tool execution telemetry."""
    prior_reasoning: list[str] = []
    markers = (*PROFILE["markers"]["reasoning"], *PROFILE["markers"]["conclusion"])
    for m in (agent_messages or [])[:-1]:
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            cleaned = m.content
            for marker in markers:
                cleaned = re.sub(re.escape(marker), "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
            if cleaned and cleaned not in prior_reasoning:
                prior_reasoning.append(cleaned)
    if prior_reasoning:
        return "\n\n".join(prior_reasoning)

    docs: list[str] = []
    tools_run: list[str] = []
    for m in agent_messages or []:
        if isinstance(m, ToolMessage):
            t_name = getattr(m, "name", "") or "tool"
            if t_name not in tools_run:
                tools_run.append(t_name)
            content_s = str(getattr(m, "content", ""))
            for doc_m in re.findall(r"Document ID:\s*([A-Za-z0-9_.-]+)", content_s):
                if doc_m not in docs and doc_m != "unknown":
                    docs.append(doc_m)
    for r in ledger or []:
        t_name = getattr(r, "tool_name", "") or "tool"
        if t_name not in tools_run:
            tools_run.append(t_name)
        content_s = str(getattr(r, "output", "") or getattr(r, "output_first_line", "") or "")
        for doc_m in re.findall(r"Document ID:\s*([A-Za-z0-9_.-]+)", content_s):
            if doc_m not in docs and doc_m != "unknown":
                docs.append(doc_m)

    doc_summary = ", ".join(docs) if docs else "indexed corpus"
    tool_summary = ", ".join(tools_run) if tools_run else "retrieval pipeline"
    return f"Executed {tool_summary} over {doc_summary}."


def enforce_response_protocol(text: str, agent_messages: list | None = None, ledger: list | None = None) -> str:
    """Guarantee 5/5 streaming protocol markers and section structure on agent outputs.

    Args:
        text: Raw synthesized agent response text.
        agent_messages: Optional sequence of LangChain messages from the active turn.
        ledger: Optional durable `ToolExecutionRecord` sequence from graph state.

    Returns:
        Protocol-compliant response containing `[REASONING]`, `[CONCLUSION]`,
        `### CONCLUSION`, `### DETAILED ANALYSIS`, and verified `**SOURCES**`.
    """
    if not text or not text.strip() or text.strip() == PROFILE["strings"]["unable_to_answer"]:
        return text

    text = sanitize_plain_text_math(text)
    text = normalize_protocol_headings(text)

    r_marker = PROFILE["markers"]["reasoning_primary"]
    c_marker = PROFILE["markers"]["conclusion_primary"]
    has_r, has_c = r_marker in text, c_marker in text

    if not has_c:
        c_idx, matched_alt_len = -1, 0
        for anchor in ("### CONCLUSION", "⚠️"):
            pos = text.find(anchor)
            if pos != -1 and (c_idx == -1 or pos < c_idx):
                c_idx, matched_alt_len = pos, 0
        if c_idx == -1:
            for alt in PROFILE["markers"]["conclusion"]:
                pos = text.upper().find(alt.upper())
                if pos != -1:
                    c_idx, matched_alt_len = pos, len(alt)
                    break

        if c_idx != -1:
            before, after = text[:c_idx].strip(), text[c_idx + matched_alt_len :].strip()
            for r_alt in PROFILE["markers"]["reasoning"]:
                if before.upper().startswith(r_alt.upper()):
                    before = before[len(r_alt) :].strip()
                    break
            before = before or _build_deterministic_reasoning(agent_messages, ledger=ledger)
            if not after.startswith(("### CONCLUSION", "⚠️")):
                after = f"### CONCLUSION\n\n{after}"
            text = f"{r_marker}\n{before}\n\n{c_marker}\n{after}"
        else:
            reasoning_block = _build_deterministic_reasoning(agent_messages, ledger=ledger)
            body = text.strip()
            for r_alt in PROFILE["markers"]["reasoning"]:
                if body.upper().startswith(r_alt.upper()):
                    body = body[len(r_alt) :].strip()
                    break
            if not body.startswith(("### CONCLUSION", "⚠️")):
                body = f"### CONCLUSION\n\n{body}"
            text = f"{r_marker}\n{reasoning_block}\n\n{c_marker}\n{body}"
    elif not has_r:
        before, _, after = text.partition(c_marker)
        for r_alt in PROFILE["markers"]["reasoning"]:
            if before.strip().upper().startswith(r_alt.upper()):
                before = before.strip()[len(r_alt) :].strip()
                break
        reasoning_block = before.strip() or _build_deterministic_reasoning(agent_messages, ledger=ledger)
        after_clean = after.strip()
        if not after_clean.startswith(("### CONCLUSION", "⚠️")):
            after_clean = f"### CONCLUSION\n\n{after_clean}"
        text = f"{r_marker}\n{reasoning_block}\n\n{c_marker}\n{after_clean}"
    else:
        head, _, after = text.partition(c_marker)
        after_clean = after.strip()
        if not after_clean.startswith(("### CONCLUSION", "⚠️")) and "Not found:" not in after_clean[:80]:
            text = f"{head.rstrip()}\n\n{c_marker}\n### CONCLUSION\n\n{after_clean}"

    head_part, sep, body_part = text.partition(c_marker)
    if sep:
        is_abstention = is_closed_abstention(body_part)
        if not is_abstention and "### CONCLUSION" not in body_part:
            body_part = f"### CONCLUSION\n\n{body_part.lstrip()}"
        if not is_abstention and "### DETAILED ANALYSIS" not in body_part:
            if "---" in body_part:
                pre_src, src_sep, post_src = body_part.partition("---")
                paragraphs = [p.strip() for p in pre_src.strip().split("\n\n") if p.strip()]
                if len(paragraphs) >= 2:
                    split_idx = 3 if (len(paragraphs) >= 3 and paragraphs[1].startswith("⚠️")) else 2
                    detail_body = "\n\n".join(paragraphs[split_idx:]) or "See quantitative and statutory verification above."
                    body_part = (
                        "\n\n".join(paragraphs[:split_idx])
                        + f"\n\n### DETAILED ANALYSIS\n\n{detail_body}\n\n{src_sep}{post_src}"
                    )
                else:
                    body_part = f"{pre_src.rstrip()}\n\n### DETAILED ANALYSIS\n\nSee quantitative and statutory verification above.\n\n{src_sep}{post_src}"
            else:
                body_part = f"{body_part.rstrip()}\n\n### DETAILED ANALYSIS\n\nSee quantitative and statutory verification above."

        if "**SOURCES**" not in body_part and "**INTERNET SOURCES**" not in body_part:
            body_part = (
                f"{body_part.rstrip()}\n\n"
                f"{extract_fallback_sources_footer(agent_messages, is_abstention=is_abstention, body_part=body_part, ledger=ledger)}"
            )

        body_part, has_ext_warn = normalize_external_source_warning(body_part, agent_messages, ledger=ledger)
        body_part = verify_sources_footer(
            body_part,
            agent_messages,
            is_abstention=is_abstention,
            has_external_source_warning=has_ext_warn,
            ledger=ledger,
        )
        text = f"{head_part}{sep}\n{body_part.lstrip()}"

    return text


def split_reasoning_and_conclusion(text: str) -> tuple[str, str]:
    """Deterministically split raw agent text into `(reasoning, conclusion)` lanes
    and strip all raw `[REASONING]` / `[CONCLUSION]` protocol tags.

    Args:
        text: Raw or protocol-enforced agent output string.

    Returns:
        Tuple of `(cleaned_reasoning, cleaned_conclusion)`.
    """
    if not text:
        return "", ""
    final_markers = PROFILE["markers"]["conclusion"]
    reasoning_markers = PROFILE["markers"]["reasoning"]
    reasoning_part, conclusion_part = "", text

    for marker in final_markers:
        idx = conclusion_part.upper().find(marker.upper())
        if idx != -1:
            reasoning_part, conclusion_part = conclusion_part[:idx], conclusion_part[idx + len(marker) :]
            break
    else:
        if any(m.upper() in conclusion_part.upper() for m in reasoning_markers):
            m_anchor = re.search(
                r"(?:^|\n)\s*(###\s*CONCLUSION\b|⚠️\s*\*?\*?(?:Not found(?:\s*\(Out of Scope\))?|Out of Scope):)",
                conclusion_part,
                flags=re.IGNORECASE,
            )
            if m_anchor:
                a_idx = m_anchor.start(1)
                reasoning_part, conclusion_part = conclusion_part[:a_idx], conclusion_part[a_idx:]

    for marker in (*reasoning_markers, *final_markers):
        pattern = re.compile(re.escape(marker), flags=re.IGNORECASE)
        reasoning_part = pattern.sub("", reasoning_part)
        conclusion_part = pattern.sub("", conclusion_part)

    return reasoning_part.strip(), conclusion_part.strip()
