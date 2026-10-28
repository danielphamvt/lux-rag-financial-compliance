"""Citation laundering blocker & Assistant-mode external warning banner guard (`citation_and_warning_guard.py`).

Blocks `[LOW_RELEVANCE_WARNING]` and unretrieved corpus citations from leaking into `**SOURCES**`,
relocates Assistant-mode external-source warning banners (`⚠️`) immediately below `### CONCLUSION`,
and manages the AI-generated analysis disclaimer using strings from `PROFILE['strings']`.
"""

from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

from config import SOURCE_NAMES
from domain_profile import PROFILE

AI_DISCLAIMER = PROFILE["strings"]["ai_disclaimer"]

_SOURCES_BLOCK_RE = re.compile(
    r"((?:---\s*\n+)?)\*\*SOURCES\*\*\s*\n(.*?)(?=\n+(?:---\s*\n+)?\*\*INTERNET SOURCES\*\*|\Z)",
    flags=re.DOTALL,
)
_DISCLAIMER_RE = re.compile(r"\n*" + re.escape(AI_DISCLAIMER).replace(r"\ ", r"\s+") + r"\s*", flags=re.IGNORECASE)


def _strings() -> dict:
    return PROFILE["strings"]


def is_closed_abstention(body_part: str) -> bool:
    """Return True only for closed-world RAG abstentions or out-of-scope refusals.

    Args:
        body_part: Conclusion body text to inspect.

    Returns:
        True if `body_part` represents a closed-world abstention or refusal.
    """
    head = re.split(r"###\s*DETAILED\s+ANALYSIS|\*\*(?:INTERNET\s+)?SOURCES\*\*", body_part, maxsplit=1, flags=re.I)[0]
    if re.search(r"⚠️\s*\*?\*?WARNING\b", head, flags=re.IGNORECASE):
        return False
    if re.search(r"\bINDETERMINATE\b", head, flags=re.IGNORECASE) and not re.search(
        r"outside\s+the\s+(?:indexed\s+|retrieved\s+)?corpus|\bmust\s+abstain\b|\babstain\s+from\b|⚠️\s*\*?\*?(?:Not\s+found(?:\s*\(Out\s+of\s+Scope\))?|Out\s+of\s+Scope):",
        head,
        flags=re.IGNORECASE,
    ):
        return False
    pattern = _strings().get("closed_abstention_pattern", "")
    return bool(pattern and re.search(pattern, head, flags=re.IGNORECASE))


def strip_conditional_disclaimer(text: str) -> str:
    """Drop the AI-generated-analysis disclaimer unless an external source participates.

    Args:
        text: Response text to inspect and clean.

    Returns:
        Cleaned response text.
    """
    has_ext = bool(re.search(r"WARNING — Data Source:|built-in knowledge|Internet Search|\*\*INTERNET SOURCES\*\*", text, flags=re.I))
    return _DISCLAIMER_RE.sub("\n", text) if not has_ext else text


def _build_doc_tokens() -> dict[str, set[str]]:
    kws = [k.lower() for k in PROFILE.get("source_keywords", [])]
    return {
        disp: {
            raw_fn.lower(),
            re.sub(r"\.(?:pdf|docx|md)$", "", raw_fn, flags=re.IGNORECASE).lower(),
            disp.lower(),
            disp.split("(")[0].strip().lower(),
            *(v for kw in kws if kw in raw_fn.lower() or kw in disp.lower() for v in (kw, kw.replace("_", "/"), kw.replace("_", " "))),
        } - {""}
        for raw_fn, disp in SOURCE_NAMES.items()
    }


def _iter_tool_outputs(agent_messages: list | None, ledger: list | None = None) -> list[tuple[str, str]]:
    """Collect `(tool_name, output_text)` pairs from `agent_messages` and `ledger`."""
    outputs = [(getattr(m, "name", "") or "", str(getattr(m, "content", "") or "")) for m in (agent_messages or []) if isinstance(m, ToolMessage)]
    for r in ledger or []:
        t_name, t_out = getattr(r, "tool_name", "") or "", str(getattr(r, "output", "") or getattr(r, "output_first_line", "") or "")
        if t_name and t_out:
            outputs.append((t_name, t_out))
    return outputs


def retrieved_high_rel_files(agent_messages: list | None, body_part: str = "", ledger: list | None = None) -> list[str]:
    """Return ordered display names of corpus files retrieved in non-warning `search_documents` results.

    Args:
        agent_messages: Sequence of LangChain messages in the active sub-agent turn.
        body_part: Optional conclusion/analysis body text for fallback token matching.
        ledger: Optional durable `ToolExecutionRecord` sequence from graph state.

    Returns:
        Ordered list of citation-eligible corpus display names.
    """
    files: list[str] = []
    warn_files: list[tuple[str, str]] = []
    for t_name, content_s in _iter_tool_outputs(agent_messages, ledger):
        if t_name == "search_documents":
            is_warn = "[LOW_RELEVANCE_WARNING:" in content_s or "[OUT_OF_CORPUS_INSTRUMENT_WARNING:" in content_s
            for line in re.findall(r"File Name:\s*([^\n]+)", content_s):
                for fn in (f.strip() for f in line.split(",") if f.strip() and f.strip() != "unknown"):
                    disp = SOURCE_NAMES.get(fn, re.sub(r"\.(?:pdf|docx|md)$", "", fn, flags=re.IGNORECASE))
                    if is_warn:
                        warn_files.append((fn, disp))
                    elif disp not in files:
                        files.append(disp)
    if not files and warn_files and body_part:
        pre_src, doc_tokens = body_part.split("**SOURCES**")[0].lower(), _build_doc_tokens()
        for raw_fn, disp in warn_files:
            if any(t and t in pre_src for t in doc_tokens.get(disp, {raw_fn.lower(), disp.lower()})) and disp not in files:
                files.append(disp)
    return files


def extract_fallback_sources_footer(
    agent_messages: list | None, is_abstention: bool = False, body_part: str = "", ledger: list | None = None
) -> str:
    """Build a deterministic `**SOURCES**` footer from retrieved `ToolMessage`s or ledger when omitted by the LLM.

    Args:
        agent_messages: Sequence of LangChain messages in the active sub-agent turn.
        is_abstention: Whether the response is a closed-world abstention.
        body_part: Conclusion/analysis body text.
        ledger: Optional durable `ToolExecutionRecord` sequence from graph state.

    Returns:
        Formatted `**SOURCES**` footer string.
    """
    files = [] if is_abstention else retrieved_high_rel_files(agent_messages, body_part=body_part, ledger=ledger)
    if not files:
        return f"---\n**SOURCES**\n{_strings()['sources_no_relevant_docs']}"
    return "---\n**SOURCES**\n" + "\n".join(f"- {f}" for f in files)


def normalize_external_source_warning(
    body_part: str, agent_messages: list | None = None, ledger: list | None = None
) -> tuple[str, bool]:
    """Relocate and normalize Assistant-mode external-source warning banners and disclaimers.

    Args:
        body_part: Conclusion/analysis body text.
        agent_messages: Optional sequence of LangChain messages in the active sub-agent turn.
        ledger: Optional durable `ToolExecutionRecord` sequence from graph state.

    Returns:
        Tuple of `(normalized_body_part, has_external_source_warning)`.
    """
    if is_closed_abstention(body_part):
        return body_part, False

    strs = _strings()
    disclaimer = strs["ai_disclaimer"]
    body_no_hdr = re.sub(r"⚠️\s*\*?\*?WARNING[^\n]*?:\*?\*?", "", body_part, flags=re.IGNORECASE)
    pre_sources_no_hdr = re.split(r"\*\*(?:INTERNET\s+)?SOURCES\*\*", body_no_hdr, maxsplit=1)[0]
    tool_outputs = _iter_tool_outputs(agent_messages, ledger)
    used_internet = any(t_name == "internet_search" or "[INTERNET_DATA_START]" in content_s for t_name, content_s in tool_outputs) or bool(
        re.search(
            r"\*\*INTERNET SOURCES\*\*|external Internet search|Internet Search \(External Web Reference\)|Data Source:[^\n]*Internet Search",
            body_part,
            flags=re.IGNORECASE,
        )
        and not re.search(r"AI's Built-in Knowledge\s*/\s*Internet Search", body_part, flags=re.IGNORECASE)
    )
    has_ooc_warn = any("[OUT_OF_CORPUS_INSTRUMENT_WARNING:" in content_s for _, content_s in tool_outputs)
    has_low_warn = any("[LOW_RELEVANCE_WARNING:" in content_s for _, content_s in tool_outputs)
    has_warn_banner = bool(re.search(r"⚠️\s*\*?\*?WARNING\b", body_part, flags=re.IGNORECASE))
    is_indeterminate = "INDETERMINATE" in body_part[:350].upper()
    used_builtin = bool(
        re.search(r"built-in knowledge|pre-trained knowledge|Data Source:\s*AI's Built-in Knowledge", pre_sources_no_hdr, flags=re.IGNORECASE)
        or (has_warn_banner and not used_internet)
        or (has_ooc_warn and not used_internet and not is_indeterminate)
    )
    valid_corpus_files = retrieved_high_rel_files(agent_messages, body_part=body_part, ledger=ledger)

    if valid_corpus_files and not used_internet and not used_builtin and not has_ooc_warn and not has_low_warn and not has_warn_banner:
        return _DISCLAIMER_RE.sub("\n\n", body_part), False
    if not has_warn_banner and not used_internet and not used_builtin:
        return body_part, False

    pre_conc_match = re.match(
        r"^\s*(⚠️\s*\*?\*?WARNING[^\n]*(?:\n(?!###\s*CONCLUSION\b)[^\n]+)*)\s*\n+(###\s*CONCLUSION\b[ \t]*\n*)",
        body_part,
        flags=re.IGNORECASE,
    )
    if pre_conc_match:
        body_part = f"### CONCLUSION\n\n{pre_conc_match.group(1).strip()}\n\n{body_part[pre_conc_match.end():].lstrip()}"
    elif has_warn_banner and "### CONCLUSION" not in body_part:
        body_part = f"### CONCLUSION\n\n{body_part.lstrip()}"

    if valid_corpus_files:
        sub_label = (
            "Internet Search"
            if (used_internet and not used_builtin)
            else ("AI's Built-in Knowledge" if (used_builtin and not used_internet) else "AI's Built-in Knowledge / Internet Search")
        )
        target_header = strs["warning_header_hybrid"].format(sub_label=sub_label)
        default_banner = strs["warning_banner_hybrid"].format(target_header=target_header)
    elif used_internet:
        target_header, default_banner = strs["warning_header_internet"], strs["warning_banner_internet"].format(target_header=strs["warning_header_internet"])
    else:
        target_header, default_banner = strs["warning_header_builtin"], strs["warning_banner_builtin"].format(target_header=strs["warning_header_builtin"])

    if has_warn_banner:
        body_part = re.sub(r"⚠️\s*\*\*WARNING(?:\s*[—–-]\s*Data Source:[^*\n]+)?:\*\*|⚠️\s*\*?\*?WARNING:?\*?\*?", target_header, body_part, count=1, flags=re.I)
        body_part = re.sub(r"(?m)^(###\s*CONCLUSION)[ \t]*\n(⚠️)", r"\1\n\n\2", body_part)
    else:
        body_part = re.sub(r"(###\s*CONCLUSION\b[ \t]*\n+)", f"### CONCLUSION\n\n{default_banner}\n\n", body_part, count=1, flags=re.I)

    src_match = re.search(r"\*\*(?:INTERNET\s+)?SOURCES\*\*", body_part)
    if src_match:
        pre_s, s_sep, post_s = body_part[: src_match.start()], src_match.group(0), _DISCLAIMER_RE.sub("\n", body_part[src_match.end() :])
        if not _DISCLAIMER_RE.search(pre_s):
            before_hr, hr_sep, after_hr = pre_s.rpartition("---")
            pre_s = f"{before_hr.rstrip()}\n\n{disclaimer}\n\n{hr_sep}{after_hr}" if hr_sep else f"{pre_s.rstrip()}\n\n{disclaimer}\n\n"
        body_part = f"{pre_s}{s_sep}{post_s}"
    elif not _DISCLAIMER_RE.search(body_part):
        body_part = f"{body_part.rstrip()}\n\n{disclaimer}"

    return body_part, True


def verify_sources_footer(
    body_part: str,
    agent_messages: list | None = None,
    is_abstention: bool = False,
    has_external_source_warning: bool = False,
    ledger: list | None = None,
) -> str:
    """Strip unretrieved or low-relevance corpus citations from `**SOURCES**` while preserving valid sources.

    Args:
        body_part: Conclusion/analysis body text containing `**SOURCES**`.
        agent_messages: Optional sequence of LangChain messages in the active sub-agent turn.
        is_abstention: Whether the response is a closed-world abstention.
        has_external_source_warning: Whether an external data-source warning is active.
        ledger: Optional durable `ToolExecutionRecord` sequence from graph state.

    Returns:
        Body text with verified `**SOURCES**` footer.
    """
    if is_abstention:
        return _SOURCES_BLOCK_RE.sub(lambda _: extract_fallback_sources_footer(agent_messages, is_abstention=True, ledger=ledger), body_part, count=1)

    tool_outputs = _iter_tool_outputs(agent_messages, ledger)
    has_retrieval_msgs = any(
        "File Name:" in content_s or "[LOW_RELEVANCE_WARNING:" in content_s or "[OUT_OF_CORPUS_INSTRUMENT_WARNING:" in content_s or t_name == "internet_search"
        for t_name, content_s in tool_outputs
    )
    if not has_retrieval_msgs and not has_external_source_warning:
        return body_part

    ordered_valid_files = retrieved_high_rel_files(agent_messages, body_part=body_part, ledger=ledger)
    valid_files = set(ordered_valid_files)
    if not valid_files and not has_external_source_warning:
        return _SOURCES_BLOCK_RE.sub(lambda _: extract_fallback_sources_footer(agent_messages, is_abstention=True, ledger=ledger), body_part, count=1)

    match = _SOURCES_BLOCK_RE.search(body_part)
    if not match:
        return body_part

    strs, doc_tokens = _strings(), _build_doc_tokens()
    kept: list[str] = []
    for ln in match.group(2).splitlines():
        stripped = ln.strip()
        if not stripped.startswith(("-", "*", "•")):
            kept.append(ln)
            continue
        if "no relevant documents" in stripped.lower():
            continue
        matched = {d for d, toks in doc_tokens.items() if any(t in stripped.lower() for t in toks)}
        if (matched and (matched & valid_files)) or (not matched and has_external_source_warning):
            kept.append(ln)

    if has_external_source_warning:
        kept_lower = "\n".join(kept).lower()
        missing_corpus = [f"- {disp}" for disp in ordered_valid_files if not any(t in kept_lower for t in doc_tokens.get(disp, {disp.lower()}))]
        if missing_corpus:
            kept = missing_corpus + [ln for ln in kept if ln.strip()]
        has_ext_line = any(
            any(tok in ln.lower() for tok in ("built-in knowledge", "http://", "https://", "internet", "external web"))
            for ln in kept if ln.strip().startswith(("-", "*", "•"))
        )
        if not has_ext_line:
            kept.append(strs["sources_builtin_knowledge"] if re.search(r"built-in knowledge", body_part, flags=re.IGNORECASE) else strs["sources_internet_search"])

    if not any(ln.strip().startswith(("-", "*", "•")) for ln in kept):
        replacement = extract_fallback_sources_footer(agent_messages, is_abstention=is_abstention, body_part=body_part, ledger=ledger)
    else:
        sep = match.group(1) or "---\n"
        replacement = f"{sep}**SOURCES**\n" + "\n".join(kept)
    return body_part[: match.start()] + replacement + body_part[match.end() :]
