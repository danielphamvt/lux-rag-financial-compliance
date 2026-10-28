"""Out-of-corpus regulatory instrument detection for search results.

Flags instruments referenced in a query that are absent from the indexed corpus,
so the agent can abstain or scope its answer instead of hallucinating citations.
"""

from __future__ import annotations

import re

from domain_profile import PROFILE


def _detect_out_of_corpus_instruments(query: str, chunks: list) -> list[str]:
    """Return deduplicated list of external regulatory instruments referenced in `query` that are not in the corpus."""
    detected: list[str] = []
    ooc_cfg = PROFILE.get("out_of_corpus_detection", {})
    circ_regex = ooc_cfg.get("circular_regex", "")
    indexed_circs = set(ooc_cfg.get("indexed_circulars", []))
    circ_prefix = ooc_cfg.get("circular_label_prefix", "Circular")
    if circ_regex:
        for match in re.finditer(circ_regex, query or "", re.IGNORECASE):
            num = match.group(1)
            if num not in indexed_circs:
                label = f"{circ_prefix} {num}"
                if label not in detected:
                    detected.append(label)

    combined_chunk_text = " ".join(str(getattr(c, "text", "") or "") for c in (chunks or [])).lower()
    for item in ooc_cfg.get("external_instruments", []):
        pat, label, chunk_tokens = item.get("pattern", ""), item.get("label", ""), item.get("tokens", [])
        if pat and label and re.search(pat, query or "", re.IGNORECASE):
            if not any(tok.lower() in combined_chunk_text for tok in chunk_tokens):
                if label not in detected:
                    detected.append(label)

    indexed_directives = {s.lower() for s in ooc_cfg.get("indexed_directives", [])}
    directive_regex = ooc_cfg.get("directive_regex", "")
    if directive_regex:
        for m_dir in re.finditer(directive_regex, query or "", re.IGNORECASE):
            dir_num = m_dir.group(1)
            if dir_num.lower() not in indexed_directives and dir_num.lower() not in combined_chunk_text:
                if not any(dir_num.lower() in d.lower() for d in detected):
                    detected.append(f"Directive {dir_num}")

    regulation_regex = ooc_cfg.get("regulation_regex", "")
    if regulation_regex:
        for m_reg in re.finditer(regulation_regex, query or "", re.IGNORECASE):
            reg_num = m_reg.group(1)
            if reg_num.lower() not in combined_chunk_text and not any(reg_num.lower() in d.lower() for d in detected):
                detected.append(f"Regulation (EU) {reg_num}")
    return detected
