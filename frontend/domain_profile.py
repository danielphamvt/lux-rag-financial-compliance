"""Domain profile loader.

A domain profile is a swappable content pack that turns the generic RAG
pipeline into a domain-specific assistant (financial compliance, EU law,
administrative documents, ...). A profile is pure data — no code execution:

- ``profiles/<name>/profile.json`` — settings: response language, streaming
  markers, greeting/small-talk handling, retrieval keywords, and
  assistant-facing runtime strings.
- ``profiles/<name>/prompts/*.md`` — one prompt template per file (file
  stem = template key), rendered with ``str.format`` placeholders.

Select the active profile with the ``RAG_PROFILE`` environment variable
(directory name under ``profiles/``). Keys missing from a profile fall back
to the generic English defaults below, so a new domain only needs to
override what it cares about — shipping only some ``prompts/*.md`` files
overrides just those templates.

This module must stay dependency-light (stdlib only, no config/LLM/embedding
imports): lightweight modules such as the routing predicates and the
internet-search skill import it directly.
"""

import copy
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
PROFILE_NAME = os.getenv("RAG_PROFILE", "financial_compliance")

_PARTIAL_RE = re.compile(r"\{\{(_[a-zA-Z0-9_]+)\}\}")
_FIELD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_COORD_SPLIT_HEADER = "## TECHNIQUE ACTIVATION TABLE"
_TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+\.md)`\s*\|\s*(.+?)\s*\|$", re.MULTILINE)
_BACKTICK_TOKEN_RE = re.compile(r"`([^`]+)`")

_DEFAULT_DIRECTIVE = (
    "Be direct, brief, and concise. Provide core information only.\n"
    "If you're unsure about the user's intent, ask for clarification rather than making assumptions."
)
_DEFAULT_COMPACT_CALC_VERIFICATION = (
    "- Tool & Arithmetic Verification: If any `calculate_decimal` line returned `ERROR`, "
    "used a wrong base/cap, or lacks a sum/difference, re-call `calculate_decimal` immediately; otherwise proceed to Step 2."
)

# Generic English fallbacks. Every consumer reads through PROFILE, which is
# the deep-merged result of the active profile over these defaults.
DEFAULT_PROFILE = {

    "name": "generic",
    "title": "Document Compliance & Audit RAG",
    "description": "Regulatory & Financial Compliance Controls",
    "response_language": "English",
    "corpus_domain_label": "the indexed compliance corpus",
    "sample_questions": [],
    "source_display_names": {},
    "document_labels": {},
    "source_keywords": [],
    "supported_share_classes": [],
    "supported_audit_types": ["nav_pricing_error", "fee_accrual_audit"],
    "bitemporal_regimes": {
        "cutoff_date": "2025-01-01",
        "default_audit_date": "2025-02-15",
        "default_fund_type": "Equity UCITS",
        "default_accrual_days": 365,
        "predecessor": {"label": "Predecessor Regime", "short_label": "Predecessor", "default_audit_date": "2024-06-15", "years": [], "tokens": []},
        "successor": {"label": "Successor Regime", "short_label": "Successor", "default_audit_date": "2025-02-15", "years": [], "tokens": []},
        "transitional": {"default_audit_date": "2025-01-10", "keywords": [], "exclude_when_tokens": []},
        "instrument_split_pattern": "",
    },
    "out_of_corpus_detection": {
        "circular_regex": "",
        "circular_label_prefix": "Circular",
        "indexed_circulars": [],
        "directive_regex": r"\bDirective\s+((?:19|20)\d{2}/\d+/(?:EC|EU|EEC))\b",
        "regulation_regex": r"\bRegulation\s+\((?:EU|EC)\)\s+(?:No\s+)?(\d{2,4}/\d+)\b",
        "indexed_directives": [],
        "external_instruments": [],
    },
    "routing": {
        "domain_signals": [
            "compliance", "regulatory", "regulation", "directive", "circular", "law", "statute", "article", "section",
            "chapter", "table", "point", "paragraph", "annex", "footnote", "audit", "auditor", "accounting", "financial",
            "finance", "fund", "portfolio", "asset", "nav", "gav", "aum", "bps", "fee", "tax", "breach", "tolerance",
            "threshold", "valuation", "pricing", "investor", "share", "unit", "class", "report", "eur", "usd",
        ],
        "domain_metric_pattern": r"(?:%|\b\d+(?:\.\d+)?\s*(?:%|bps|eur|usd|gbp|chf|jpy|million|m\b|k\b))",
        "out_of_domain_triggers": [
            "recipe", "cook", "cooking", "bake", "baking", "ingredient", "pasta", "pizza", "chicken", "soup", "cake",
            "salad", "pho", "spaghetti", "calories", "workout", "gym", "exercise", "yoga", "symptom", "headache",
            "medicine", "doctor", "hospital", "football", "soccer", "basketball", "tennis", "cricket", "golf",
            "olympics", r"world\s+cup", r"champions\s+league", r"premier\s+league", "nba", "nfl", "fifa", "uefa",
            "messi", "ronaldo", "lebron", "weather", "forecast", "rain", "sunny", "temperature", "celsius", "fahrenheit",
            "umbrella", "poem", "poetry", "haiku", "sonnet", "limerick", "lyrics", "chord", "guitar", "piano", "movie",
            "actor", "actress", "hollywood", "netflix", "anime", "manga", "horoscope", "astrology", "zodiac", "tarot",
            "flight", "hotel", "vacation", "tourist", "sightseeing",
            r"write\s+a\s+(?:python|javascript|typescript|java|c\+\+|rust|go|sql|html|css|react|bash)\b",
            r"(?:python|javascript|typescript|java|c\+\+|rust)\s+(?:function|script|program)\b",
            "leetcode", r"binary\s+tree", "quicksort", "mergesort", "fibonacci", "palindrome", "fizzbuzz",
            r"linked\s+list", "dijkstra", "dockerfile", r"kubernetes\s+pod", r"css\s+flexbox", r"react\s+hook",
            r"tell\s+me\s+a\s+joke", r"bedtime\s+story", r"love\s+letter",
        ],
    },
    "harness_config": {
        "max_ast_exponent": 1000,
        "max_precision_digits": 24,
        "currencies": ["EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD", "SEK", "NOK", "DKK", "SGD", "HKD", "CNY"],
        "units": ["bps", "shares", "share", "units", "unit", "days", "day", "months", "month", "years", "year", r"per\s+annum", r"p\.?a\.?"],
    },
    "greetings": ["hi", "hello", "hey", "thank you", "thanks", "bye", "goodbye"],
    "small_talk": {
        "thanks_triggers": ["thank you", "thanks"],
        "farewell_triggers": ["bye", "goodbye"],
        "thanks_response": "You're welcome! Is there anything else I can help with?",
        "farewell_response": "Goodbye! See you next time.",
        "default_response": "Hello! How can I help you today?",
    },
    "internet_search": {"query_suffixes": []},
    "markers": {
        "reasoning_primary": "[REASONING]",
        "conclusion_primary": "[CONCLUSION]",
        "reasoning": ["[REASONING]", "REASONING:", "**REASONING:**", "**REASONING**"],
        "conclusion": ["[CONCLUSION]", "CONCLUSION:", "**CONCLUSION:**", "**CONCLUSION**", "ANSWER:", "**ANSWER:**", "FINAL ANSWER:"],
        "reasoning_closing": '<div style="margin-top: 8px; line-height: 1.6;">End of reasoning. Now I will provide the answer to the user...</div>',
    },
    "strings": {
        "status_analyzing": "Analyzing your question...",
        "status_retrieving": "Retrieving and cross-checking documents...",
        "status_thinking": "Thinking...",
        "status_answering": "Writing the answer...",
        "status_processing": "Processing...",
        "error_notice": (
            '<div style="padding: 12px 16px; border: 1px solid rgba(239, 68, 68, 0.5); '
            'border-radius: 8px; background: rgba(239, 68, 68, 0.08); line-height: 1.6;">'
            "⚠️ <b>The system encountered an error while processing this question.</b> "
            "Please try again later, or check the connection to the LLM provider (Ollama/API key).</div>"
        ),
        "retrieval_no_results": "No documents matching the query were found in the database.",
        "forced_tool_empty_query": "Please enter your question after the command.",
        "out_of_scope_notice": "### CONCLUSION\n\n⚠️ **Out of Scope:** Your question falls outside the configured compliance and audit domain.",
        "no_answers_generated": "No answers were generated.",
        "unable_to_answer": "Unable to generate an answer.",
        "no_query_fallback": "No specific question available; please summarize based on the data below.",
        "internet_no_results": "No results were found on the internet.",
        "internet_timeout": "Internet search timed out. Please try again later.",
        "internet_error": "Internet search error: {error}",
        "ai_disclaimer": "Note: This is an AI-generated analysis, not professional advice. Please consult a qualified professional before making a decision.",
        "sources_no_relevant_docs": "- No relevant documents were found in the document database.",
        "sources_builtin_knowledge": "- Not found in the provided document database; the answer is based on the AI's built-in knowledge.",
        "sources_internet_search": "- External Web Reference (Internet Search)",
        "warning_header_hybrid": "⚠️ **WARNING — Data Source: Hybrid (Internal Compliance Corpus + {sub_label}):**",
        "warning_banner_hybrid": "{target_header} This answer combines provisions from the internal compliance database with external information. External portions are provided for reference only.",
        "warning_header_internet": "⚠️ **WARNING — Data Source: Internet Search (External Web Reference):**",
        "warning_banner_internet": "{target_header} No matching provision was found in the internal compliance database. This answer relies on external Internet search results and is NOT certified from the internal audit corpus.",
        "warning_header_builtin": "⚠️ **WARNING — Data Source: AI's Built-in Knowledge:**",
        "warning_banner_builtin": "{target_header} I could not find information related to your question in the provided document database. This answer relies on the AI's built-in knowledge and is provided for reference only.",
        "closed_abstention_notice": "⚠️ **Not found (Out of Scope):** The queried framework is out of scope and not governed by {corpus_domain_label}, as no matching provisions were found in the provided document database.",
        "closed_abstention_pattern": r"(?:⚠️\s*\*?\*?(?:Not\s+found(?:\s*\(Out\s+of\s+Scope\))?|Out\s+of\s+Scope):|(?<!warning:\s)\bnot\s+found(?:\s*\(out\s+of\s+scope\))?:|not\s+covered|not\s+governed\s+by\b|outside\s+the\s+(?:indexed\s+|retrieved\s+)?corpus|\bmust\s+abstain\b|\babstain\s+from\b|no\s+retrieved\s+documents?\s+(?:mention|cite|contain)|not\s+present\s+in\s+the\s+retrieved\s+documents)",
    },
    # Single Source of Truth for active prompts: frontend/profiles/<profile_name>/prompts/*.md
    "partials": {
        "_immutable_directive": _DEFAULT_DIRECTIVE,
        "_compact_calc_verification": _DEFAULT_COMPACT_CALC_VERIFICATION,
    },
    "prompts": {
        "conversation_summary": "Summarize the conversation briefly in {response_language}.",
        "rewrite_query": "Rewrite the user query into a clear, self-contained search query.",
        "orchestrator": f"You are a document research assistant.\n{{knowledge_rule}} Always reply in {{response_language}}.\n\n{_DEFAULT_DIRECTIVE}",
        "stage1_retrieval_plan": "Call `{tool_name_to_force}` first. Optional intent under `{reasoning_primary}`. Do not emit `{conclusion_primary}` yet.",
        "stage2_calculation_gate": "Call `calculate_decimal` for all quantitative calculations before emitting `{conclusion_primary}`. {active_calculation_techniques}",
        "stage3_audit_synthesis": "Synthesize retrieved evidence using `{reasoning_primary}` then `{conclusion_primary}`. {sources_fallback} Reply in {response_language}.",
        "orchestrator_knowledge_open": "Ground answers in retrieved documents; warn after {conclusion_primary} if using built-in knowledge.",
        "orchestrator_sources_open": "If using built-in knowledge, list built-in knowledge under Sources.",
        "orchestrator_knowledge_rag_only": "Answer strictly from retrieved documents; if not found, state so after {conclusion_primary}.",
        "orchestrator_sources_rag_only": "If no relevant documents were found, state so under Sources.",
        "fallback_response": "Synthesize the available research context. {source_integrity_rule} {sources_warning} Reply in {response_language}.",
        "fallback_rules_open": "Use retrieved context first; warn if falling back to built-in knowledge.",
        "fallback_warning_open": "Include a warning banner when falling back to built-in knowledge.",
        "fallback_rules_rag_only": "Use retrieved context exclusively; abstain if not found.",
        "fallback_warning_rag_only": "List that no relevant documents were found under Sources.",
        "context_compression": "Compress the retrieved research context into structured Markdown findings and gaps.",
        "internet_search_answer": "Answer `{question}` using only `{context}`.{hist_section} Reply in {response_language}.",
        "internet_no_results_answer": "No internet results found for `{question}`.",
        "clarification_fallback": "Your question is not clear enough. Please provide more information to help me find the documents for you.",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (dicts merged, leaves replaced)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _compile_partials(prompts: dict, partials: dict) -> dict:
    """Expand {{_partial_name}} placeholders at profile load time."""
    if not partials:
        return prompts

    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in partials:
            return partials[key]
        logger.warning("Unknown prompt partial '{{%s}}' left unexpanded.", key)
        return match.group(0)

    return {name: _PARTIAL_RE.sub(_replace, template) for name, template in prompts.items()}


def _load_calculation_techniques(prompts_dir: Path, partials: dict) -> list[dict]:
    """Load coordinator rules and technique routing table from prompts/02_calculation_techniques/.

    Args:
        prompts_dir: Path to the profile's prompts directory.
        partials: Dict of partial templates updated in-place with '_calculation_coordinator'.

    Returns:
        List of technique dicts with keys 'file', 'triggers', and 'content'.
    """
    tech_dir = prompts_dir / "02_calculation_techniques"
    if not tech_dir.is_dir():
        tech_dir = prompts_dir / "calculation_techniques"
    coord_path = tech_dir / "coordinator.md"
    if not coord_path.is_file():
        return []

    raw_coord = coord_path.read_text(encoding="utf-8")
    parts = raw_coord.split(_COORD_SPLIT_HEADER, 1)
    partials["_calculation_coordinator"] = parts[0].strip()
    if len(parts) < 2:
        return []

    techniques: list[dict] = []
    for match in _TABLE_ROW_RE.finditer(parts[1]):
        filename, trigger_col = match.group(1).strip(), match.group(2)
        tech_path = tech_dir / filename
        if not tech_path.is_file():
            continue
        triggers = [t.strip().lower() for t in _BACKTICK_TOKEN_RE.findall(trigger_col) if t.strip()]
        techniques.append({"file": filename, "triggers": triggers, "content": tech_path.read_text(encoding="utf-8").strip()})
    return techniques


def _load_profile() -> dict:
    profile_dir = PROFILES_DIR / PROFILE_NAME
    manifest_path = profile_dir / "profile.json"
    if not manifest_path.exists():
        logger.warning("Profile '%s' not found at %s — falling back to generic defaults.", PROFILE_NAME, manifest_path)
        return copy.deepcopy(DEFAULT_PROFILE)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read profile manifest %s (%s) — falling back to generic defaults.", manifest_path, e)
        return copy.deepcopy(DEFAULT_PROFILE)

    # Recursively scan hierarchical prompt directories (00_..06_).
    prompts_dir = profile_dir / "prompts"
    if prompts_dir.is_dir():
        partials = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(prompts_dir.rglob("_*.md")) if p.is_file()}
        profile["calculation_techniques"] = _load_calculation_techniques(prompts_dir, partials)
        profile["partials"] = partials
        tech_dirs = {prompts_dir / "02_calculation_techniques", prompts_dir / "calculation_techniques"}
        raw_prompts = {
            p.stem: p.read_text(encoding="utf-8")
            for p in sorted(prompts_dir.rglob("*.md"))
            if p.is_file() and not p.name.lower().startswith(("readme", "_")) and p.parent not in tech_dirs
        }
        profile.setdefault("prompts", {}).update(_compile_partials(raw_prompts, partials))

    if profile.get("name") != PROFILE_NAME:
        logger.warning("Profile directory '%s' declares name %r — continuing anyway.", PROFILE_NAME, profile.get("name"))
    return _deep_merge(DEFAULT_PROFILE, profile)


def _prompts_max_mtime(profile_name: str) -> int:
    prompts_dir = PROFILES_DIR / profile_name / "prompts"
    return max((p.stat().st_mtime_ns for p in prompts_dir.rglob("*.md")), default=0) if prompts_dir.is_dir() else 0


PROFILE = _load_profile()
_PROFILE_MTIME = _prompts_max_mtime(PROFILE_NAME)


def select_calculation_techniques(query_context: str = "", max_techniques: int = 1) -> str:
    """Activate matching calculation techniques from prompts/calculation_techniques/coordinator.md.

    Args:
        query_context: User query and/or retrieved evidence text to match against triggers.
        max_techniques: Maximum number of technique playbooks to inject on a single turn.

    Returns:
        Concatenated markdown blocks for the highest-scoring matched techniques, or empty string.
    """
    if not query_context or not query_context.strip():
        return ""
    lowered = query_context.lower()
    scored = [(sum(1 for trig in tech.get("triggers", []) if trig in lowered), -idx, tech["content"]) for idx, tech in enumerate(PROFILE.get("calculation_techniques", []))]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(reverse=True)
    return "\n".join(content for _, _, content in scored[:max_techniques])


def get_prompt(name: str, **fields) -> str:
    """Render a prompt template from the active profile safely.

    Marker and language placeholders are always provided; callers pass any
    remaining fields (context, question, variant fragments, ...). Unknown brace
    expressions (e.g. LaTeX or JSON literals) are preserved verbatim instead of
    raising KeyError.
    """
    global _PROFILE_MTIME
    current_mtime = _prompts_max_mtime(PROFILE_NAME)
    if current_mtime > _PROFILE_MTIME:
        fresh = _load_profile()
        PROFILE["prompts"].update(fresh.get("prompts", {}))
        PROFILE["calculation_techniques"] = fresh.get("calculation_techniques", [])
        PROFILE["partials"] = fresh.get("partials", {})
        _PROFILE_MTIME = current_mtime

    default_calc_gate = PROFILE.get("partials", {}).get("_calculation_coordinator", "")
    doc_labels = list((PROFILE.get("document_labels") or PROFILE.get("source_display_names") or {}).values())
    corpus_scope = ", ".join(doc_labels) if doc_labels else "the indexed compliance corpus"
    merged = {
        "reasoning_primary": PROFILE["markers"]["reasoning_primary"],
        "conclusion_primary": PROFILE["markers"]["conclusion_primary"],
        "response_language": PROFILE["response_language"],
        "corpus_domain_label": PROFILE.get("corpus_domain_label", "the indexed compliance corpus"),
        "corpus_scope": corpus_scope,
        "calculation_gate_section": default_calc_gate,
        "active_calculation_techniques": "",
        **fields,
    }
    rendered = _FIELD_RE.sub(lambda m: str(merged[m.group(1)]) if m.group(1) in merged else m.group(0), PROFILE["prompts"][name])
    return re.sub(r"\n{3,}", "\n\n", rendered)
