"""Pure routing predicates for the agent graph.

Kept free of heavy imports (config, LLM, embeddings) so they can be
unit-tested without loading models or requiring a running provider. The
greeting set and domain/off-topic routing patterns come from the active
domain profile, which is also light.
"""
import re
import unicodedata

from domain_profile import PROFILE

# Small-talk messages answered directly without retrieval.
# Matched against the FULL normalized message, never as a substring:
# short domain questions often contain "hi" inside other words
# (e.g. "hi" in "which", "this", NFD-decomposed words) and must not be misrouted.
GREETINGS = frozenset(PROFILE["greetings"])

_PUNCT_RE = re.compile(r"^[\s!.,?~…]+|[\s!.,?~…]+$")
_SLASH_CMD_RE = re.compile(r"^/(?:all_sources|internet_search|search_documents)\b\s*", re.IGNORECASE)


def _compile_domain_signal_re() -> re.Pattern:
    routing_cfg = PROFILE.get("routing", {})
    signals = routing_cfg.get("domain_signals", [])
    metric_pat = routing_cfg.get("domain_metric_pattern", "")
    parts: list[str] = []
    if signals:
        parts.append(r"\b(?:" + "|".join(signals) + r")\b")
    if metric_pat:
        parts.append(metric_pat)
    pattern = "|".join(parts) if parts else r"(?!x)x"
    return re.compile(pattern, re.IGNORECASE)


def _compile_out_of_domain_re() -> re.Pattern:
    routing_cfg = PROFILE.get("routing", {})
    triggers = routing_cfg.get("out_of_domain_triggers", [])
    pattern = r"\b(?:" + "|".join(triggers) + r")\b" if triggers else r"(?!x)x"
    return re.compile(pattern, re.IGNORECASE)


_DOMAIN_SIGNAL_RE = _compile_domain_signal_re()
_OUT_OF_DOMAIN_RE = _compile_out_of_domain_re()


def normalize_message(text: str) -> str:
    """Lowercase, NFC-normalize (macOS NFD input) and trim punctuation."""
    text = unicodedata.normalize("NFC", text or "").strip().lower()
    return _PUNCT_RE.sub("", text)


def is_greeting_message(text: str) -> bool:
    """True when the whole message is a greeting, thanks or farewell."""
    return normalize_message(text) in GREETINGS


def is_out_of_domain_message(text: str) -> bool:
    """True ONLY when message clearly asks about a non-financial off-topic subject and contains zero domain signals."""
    stripped = _SLASH_CMD_RE.sub("", text or "")
    norm = normalize_message(stripped)
    if not norm or is_greeting_message(stripped):
        return False
    if _DOMAIN_SIGNAL_RE.search(norm):
        return False
    return bool(_OUT_OF_DOMAIN_RE.search(norm))
