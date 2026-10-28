"""Math & syntax normalization harness (`latex_to_unicode_normalizer.py`).

Recursive macro/delimiter compiler that strips nested LaTeX math into plain-text Unicode.
"""

from __future__ import annotations

import re

from domain_profile import PROFILE

_CFG = PROFILE.get("harness_config", {})
_CURRENCIES = [re.escape(c) for c in (_CFG.get("currencies") or ["EUR", "USD", "GBP", "CHF"])]
_UNITS = [u for u in (_CFG.get("units") or ["bps"]) if u.isalpha()]
_CURR_OR_UNIT_ALT = "|".join([*_CURRENCIES, *_UNITS])
_CURR_ALT = "|".join(_CURRENCIES)
_DIGIT_UNIT_SPACE_RE = re.compile(rf"(\d)\s+(?:,\s+)?({_CURR_OR_UNIT_ALT})\b")
_CURR_DIGIT_SPACE_RE = re.compile(rf"\b({_CURR_ALT})\s{{2,}}(\d)")

_LATEX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\times", "×"),
    (r"\cdot", "×"),
    (r"\ast", "×"),
    (r"\div", "÷"),
    (r"\pm", "±"),
    (r"\mp", "∓"),
    (r"\leqslant", "<="),
    (r"\leq", "<="),
    (r"\le", "<="),
    (r"\geqslant", ">="),
    (r"\geq", ">="),
    (r"\ge", ">="),
    (r"\neq", "!="),
    (r"\ne", "!="),
    (r"\approx", "≈"),
    (r"\simeq", "≈"),
    (r"\cong", "≈"),
    (r"\equiv", "≡"),
    (r"\sim", "~"),
    (r"\longrightarrow", "->"),
    (r"\rightarrow", "->"),
    (r"\to", "->"),
    (r"\Longrightarrow", "=>"),
    (r"\Rightarrow", "=>"),
    (r"\implies", "=>"),
    (r"\Longleftrightarrow", "<=>"),
    (r"\Leftrightarrow", "<=>"),
    (r"\iff", "<=>"),
    (r"\infty", "∞"),
    (r"\left", ""),
    (r"\right", ""),
    (r"\bigl", ""),
    (r"\bigr", ""),
    (r"\Bigl", ""),
    (r"\Bigr", ""),
    (r"\biggl", ""),
    (r"\biggr", ""),
    (r"\Biggl", ""),
    (r"\Biggr", ""),
    (r"\bigg", ""),
    (r"\Bigg", ""),
    (r"\big", ""),
    (r"\Big", ""),
    (r"\displaystyle", ""),
    (r"\textstyle", ""),
    (r"\scriptstyle", ""),
    (r"\scriptscriptstyle", ""),
    (r"\limits", ""),
    (r"\nolimits", ""),
    (r"\quad", " "),
    (r"\qquad", " "),
    (r"\enspace", " "),
    (r"\thinspace", " "),
    (r"\,", " "),
    (r"\;", " "),
    (r"\:", " "),
    (r"\!", ""),
    (r"\%", "%"),
    (r"\{", "{"),
    (r"\}", "}"),
)


def sanitize_plain_text_math(text: str) -> str:
    """Deterministically compile/strip all LaTeX math macros and delimiters into plain-text Unicode."""
    if not text or ("\\" not in text and "$$" not in text and "$" not in text and "×" not in text and "=" not in text):
        return text

    # 1. Strip display and inline math delimiters: \[ ... \], \( ... \), $$ ... $$
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$\$\s*(.*?)\s*\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(
        r"(?<![\\$\d])\$([^\$\n]*?(?:\\[a-zA-Z]+|[×÷=]|(?:\d\s*[\+\-\*/]\s*\d))[^\$\n]*?)\$(?![\d])",
        r"\1",
        text,
    )

    # 2. Strip LaTeX environments (\begin{align*}, \end{equation}, etc.)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "", text)

    # 3. Bottom-up recursive unwrapping of nested macros (up to 8 passes)
    for _ in range(8):
        prev = text
        text = re.sub(
            r"\\(?:text|textbf|textit|textrm|texttt|mathrm|mathbf|mathit|mathsf|mathtt|mathcal|operatorname|mbox|hbox|underline|overline|emph)\{([^{}]*)\}",
            r"\1",
            text,
        )
        text = re.sub(r"\\sqrt\[([^\[\]]+)\]\{([^{}]+)\}", r"root(\1, \2)", text)
        text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
        text = re.sub(r"\\(?:d|t|c)?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1 / \2)", text)
        text = re.sub(r"([_^])\{([^{}]+)\}", r"\1\2", text)
        if text == prev:
            break

    # 4. Replace LaTeX operators, relations, spacing, and sizing macros
    for src, dst in _LATEX_REPLACEMENTS:
        text = text.replace(src, dst)

    # 5. Final catch-all sweep: unwrap any remaining \macro{...} and strip stray \macro sequences
    for _ in range(3):
        cleaned = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
        if cleaned == text:
            break
        text = cleaned
    text = re.sub(r"\\[a-zA-Z]{2,}\b\s*", "", text)
    text = re.sub(r"\\\\\s*", "\n", text)
    text = text.replace("&=", "=")

    # 6. Normalize spacing around currency/units and outer parens left by stripped delimiters
    text = _DIGIT_UNIT_SPACE_RE.sub(r"\1 \2", text)
    text = _CURR_DIGIT_SPACE_RE.sub(r"\1 \2", text)
    text = re.sub(r"\(\s+([^\n]*?[×=][^\n]*?)\s+\)", r"\1", text)
    return text
