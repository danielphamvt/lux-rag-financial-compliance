"""Gradio rendering helpers: profile strings, HTML templates, icons, and display formatting.

Pure presentation transforms used by the chat stream handler and the app shell.
"""

from __future__ import annotations

from pathlib import Path

from config import SOURCE_NAMES
from domain_profile import PROFILE
from rag_agent.harness import (
    normalize_protocol_headings,
    sanitize_plain_text_math,
    split_reasoning_and_conclusion,
    strip_conditional_disclaimer,
)
from rag_agent.nodes_edges import content_to_text

UI_DIR = Path(__file__).resolve().parent

# Runtime strings and streaming markers come from the active domain profile
STRINGS = PROFILE["strings"]
MARKERS = PROFILE["markers"]

# Load HTML templates from ui/templates
with open(UI_DIR / "templates" / "spinner.html", "r", encoding="utf-8") as f:
    SPINNER_TEMPLATE = f.read()

with open(UI_DIR / "templates" / "reasoning.html", "r", encoding="utf-8") as f:
    REASONING_TEMPLATE = f.read()

REASONING_SPIN_ICON = (
    '<svg class="reasoning-spin-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" '
    'stroke="#64748b" stroke-width="2.5" aria-hidden="true">'
    '<circle cx="12" cy="12" r="9" stroke="#64748b" stroke-opacity="0.25"></circle>'
    '<circle cx="12" cy="12" r="9" stroke="#64748b" stroke-linecap="round" stroke-dasharray="16 40"></circle>'
    '</svg>'
)

REASONING_DONE_ICON = (
    '<svg class="reasoning-done-icon" width="13" height="13" viewBox="0 0 16 16" fill="none" '
    'stroke="#64748b" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M3 8.5L6.5 12L13 5"></path>'
    '</svg>'
)

import html
import re
import time


WARNING_TRIANGLE_SVG = (
    '<span class="warning-triangle-icon" aria-hidden="true">'
    '<svg viewBox="0 0 24 24" width="16" height="16" fill="#f59e0b" stroke="#b45309" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" fill="#f59e0b" stroke="#b45309"></path>'
    '<line x1="12" y1="9" x2="12" y2="13" stroke="#ffffff" stroke-width="2.2"></line>'
    '<line x1="12" y1="17" x2="12.01" y2="17" stroke="#ffffff" stroke-width="2.5"></line>'
    '</svg></span>'
)


def decorate_warning_banners(text: str) -> str:
    """Prepend a yellow SVG warning triangle badge before external data-source warnings."""
    if not text or "⚠️ **WARNING — Data Source:" not in text:
        return text
    if "warning-triangle-icon" in text:
        return text
    return text.replace(
        "⚠️ **WARNING — Data Source:",
        f"{WARNING_TRIANGLE_SVG} ⚠️ **WARNING — Data Source:",
    )


def clean_display_sources(text: str) -> str:
    if not text:
        return text
    # SOURCE_NAMES already holds both NFC and NFD keys for every file name
    for raw_name, display_name in SOURCE_NAMES.items():
        text = text.replace(raw_name, display_name)
    # Strip trailing docx/pdf extensions case-insensitively
    text = re.sub(r'\.(docx|pdf)\b', '', text, flags=re.IGNORECASE)
    # Hard Barrier 2.1 — Delegate LaTeX-to-Unicode sanitization to the global recursive sanitizer
    text = sanitize_plain_text_math(text)
    # Normalize section headings to uppercase without trailing colon via shared harness helper
    text = normalize_protocol_headings(text)
    # Retain the AI-generated analysis disclaimer whenever any external source participates
    return strip_conditional_disclaimer(text)


def close_open_divs(html: str) -> str:
    """Close any unclosed <div> tags (streams can be cut mid-tag)."""
    deficit = html.count("<div") - html.count("</div")
    return html + "</div>" * deficit if deficit > 0 else html


# Marker words the model uses to label reasoning vs conclusion output;
# accepted synonyms for parsing, in priority order
FINAL_MARKERS = MARKERS["conclusion"]
REASONING_MARKERS = MARKERS["reasoning"]

# Line appended when the reasoning block closes and the answer begins
REASONING_CLOSING_LINE = MARKERS["reasoning_closing"]

# Friendly error notice shown when the pipeline fails (instead of a traceback)
ERROR_NOTICE = STRINGS["error_notice"]


def finalize_reasoning(reasoning_html: str) -> str:
    """Close dangling <div> tags, then append the reasoning closing line."""
    return close_open_divs(reasoning_html) + REASONING_CLOSING_LINE


def ensure_conclusion_heading(final_text: str) -> str:
    """Ensure the final answer starts with an uppercase Conclusion heading."""
    if not final_text or not final_text.strip():
        return final_text
    cleaned = re.sub(r"^\s*#{1,4}\s*\n+", "", final_text)
    stripped = cleaned.lstrip()
    if stripped.startswith("⚠️") or stripped.startswith("<div"):
        return cleaned
    if re.match(r"^(?:#{1,4}\s*|\*\*)?conclusion\b", stripped, flags=re.IGNORECASE):
        return cleaned
    if len(stripped) > 25:
        return f"### CONCLUSION\n\n{stripped}"
    return cleaned


def format_rag_telemetry_html(artifact: dict) -> str:
    """Render objective RAG execution telemetry as a minimal, harmonious card inside the Reasoning box."""
    backend = html.escape(str(artifact.get("backend", "vector_chunking")))
    latency_ms = int(artifact.get("latency_ms", 0))
    latency_str = f"{latency_ms} ms" if latency_ms < 10000 else f"{latency_ms / 1000:.2f} s"
    chunks = artifact.get("chunks") or []
    chunk_count = int(artifact.get("chunk_count", len(chunks)))
    chunk_word = "chunk" if chunk_count == 1 else "chunks"

    query = html.escape(str(artifact.get("query", "")).replace("\n", " ").strip())
    sub_queries = [
        html.escape(str(sq).replace("\n", " ").strip())
        for sq in (artifact.get("sub_queries") or [])
        if str(sq).strip()
    ]
    filter_regime = html.escape(str(artifact.get("filter_regime", "audit_date = None (All regimes)")))

    rows = [
        f'<div class="reasoning-rag-row">'
        f'<span class="reasoning-rag-label">Executed Query:</span> '
        f'<span class="reasoning-rag-val">&quot;{query}&quot;</span></div>'
    ]
    if sub_queries:
        sq_joined = " · ".join(f"&quot;{sq}&quot;" for sq in sub_queries)
        rows.append(
            f'<div class="reasoning-rag-row">'
            f'<span class="reasoning-rag-label">Sub-Queries ({len(sub_queries)}):</span> '
            f'<span class="reasoning-rag-val">{sq_joined}</span></div>'
        )
    rows.append(
        f'<div class="reasoning-rag-row">'
        f'<span class="reasoning-rag-label">Filter / Regime:</span> '
        f'<span class="reasoning-rag-val">{filter_regime}</span></div>'
    )

    if chunks:
        chunk_items = []
        for c in chunks:
            score = float(c.get("score", 0.0))
            doc_label = html.escape(str(c.get("doc_label", "Document")))
            loc = html.escape(str(c.get("location", "")))
            loc_suffix = f" — {loc}" if loc else ""
            chunk_items.append(
                f'<div class="reasoning-rag-chunk">'
                f'<span class="reasoning-rag-score">[Score: {score:.2f}]</span>'
                f'<span class="reasoning-rag-doc">{doc_label}{loc_suffix}</span>'
                f'</div>'
            )
        rows.append(
            f'<div class="reasoning-rag-row">'
            f'<span class="reasoning-rag-label">Top Evidence Chunks:</span></div>'
            f'<div class="reasoning-rag-chunks">{"".join(chunk_items)}</div>'
        )

    return (
        f'<div class="reasoning-rag-card">'
        f'<div class="reasoning-rag-header">'
        f'<span>📚 RAG Retrieval</span><span class="reasoning-rag-sep">•</span>'
        f'<code>{backend}</code><span class="reasoning-rag-sep">•</span>'
        f'<span>{latency_str}</span><span class="reasoning-rag-sep">•</span>'
        f'<span>{chunk_count} {chunk_word} retrieved</span>'
        f'</div>'
        f'<div class="reasoning-rag-body">{"".join(rows)}</div>'
        f'</div>'
    )


def format_reasoning_html(reasoning_text: str) -> str:
    """Format reasoning markdown into clean, styled HTML with bullets and muted steps."""
    if not reasoning_text or not reasoning_text.strip():
        return ""

    lines = reasoning_text.replace("\r\n", "\n").split("\n")
    formatted_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # If it's already an HTML block (like tool pill or closing note)
        if line.startswith("<div"):
            formatted_lines.append(line)
            continue

        # Format inline backticks and strip bold tags (reasoning is italic-only)
        formatted_inline = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
        formatted_inline = re.sub(r"\*\*(.+?)\*\*", r"\1", formatted_inline)
        formatted_inline = re.sub(r"__(.+?)__", r"\1", formatted_inline)
        formatted_inline = re.sub(r"<\/?(?:strong|b|h\d)>", "", formatted_inline)

        # Check for bullet points: - item, * item, • item
        bullet_match = re.match(r"^[-*•]\s+(.*)$", formatted_inline)
        # Check for numbered step header: e.g. 1. Table 10 Analysis:
        step_match = re.match(r"^(\d+\.)\s+(.*)$", formatted_inline)

        if bullet_match:
            b_text = bullet_match.group(1)
            formatted_lines.append(
                f'<div class="reasoning-bullet"><span class="bullet-dot">•</span><span class="bullet-text">{b_text}</span></div>'
            )
        elif step_match:
            s_num = step_match.group(1)
            s_text = step_match.group(2)
            formatted_lines.append(
                f'<div class="reasoning-step-header"><span class="step-num">{s_num}</span> {s_text}</div>'
            )
        else:
            if "End of reasoning" in formatted_inline:
                formatted_lines.append(f'<div class="reasoning-closing-note">{formatted_inline}</div>')
            else:
                formatted_lines.append(f'<div class="reasoning-line">{formatted_inline}</div>')

    res = "\n".join(formatted_lines)
    return close_open_divs(res)


def get_display_html(reasoning_text, final_text, phase, waiting_text="", is_completed=False, elapsed_s=None, start_ms=None):
    html = ""
    # 1. Reasoning box
    if reasoning_text:
        closed_reasoning = format_reasoning_html(reasoning_text)
        is_reasoning_done = (phase == "final") or is_completed or bool(final_text and final_text.strip())
        status_icon = REASONING_DONE_ICON if is_reasoning_done else REASONING_SPIN_ICON
        timer_str = f"{elapsed_s:.1f}s" if elapsed_s is not None else ""
        data_attrs = (
            f' data-start-ms="{start_ms}" data-elapsed="{timer_str}"'
            if start_ms is not None
            else (f' data-elapsed="{timer_str}"' if timer_str else "")
        )
        open_attr = (
            f'class="reasoning-box is-complete" data-status="done"{data_attrs}'
            if is_reasoning_done
            else f'class="reasoning-box is-active" open data-status="thinking"{data_attrs}'
        )
        rendered_box = REASONING_TEMPLATE.format(open_attr=open_attr, reasoning_content=closed_reasoning, status_icon=status_icon)
        if timer_str:
            rendered_box = rendered_box.replace(
                '<span class="reasoning-timer"></span>',
                f'<span class="reasoning-timer">{timer_str}</span>',
            )
        html += rendered_box
            
    # 2. Final answer
    has_final = bool(final_text and final_text.strip())
    if has_final:
        if html:
            html += "\n\n"
        html += decorate_warning_banners(ensure_conclusion_heading(final_text))
        
    # 3. Spinner: only display before the final answer begins streaming
    if not is_completed and not has_final:
        if html:
            html += "\n\n"
        status = waiting_text if waiting_text else (STRINGS["status_answering"] if phase == "final" else STRINGS["status_processing"])
        html += SPINNER_TEMPLATE.format(status=status)

    return html
