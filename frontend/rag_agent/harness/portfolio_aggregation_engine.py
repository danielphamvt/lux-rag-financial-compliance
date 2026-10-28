"""Symbolic `Decimal` (`ROUND_HALF_UP`) expression evaluator (`portfolio_aggregation_engine.py`).

Evaluates labeled arithmetic expressions in exact Python `Decimal` (`ROUND_HALF_UP`)
and resolves cross-line variable/label references without mutating agent formulas
or injecting unrequested calculations.
"""

from __future__ import annotations

import ast
import re
from decimal import ROUND_HALF_UP, Decimal

from rag_agent.harness.ast_decimal_evaluator import (
    collect_top_level_addends,
    eval_decimal_node,
    normalize_calc_expression,
)
from rag_agent.harness.multi_turn_calc_continuity import split_label_and_expr
from rag_engine.core.nav_audit_schemas import round_currency


def _fmt_line(lbl: str, expr: str, r2: Decimal, r4_str: str) -> str:
    return f"- {f'{lbl}: ' if lbl else ''}`{expr}` => **{r2:,.2f}** (exact 4-dec: `{r4_str}`)"


def evaluate_decimal_expressions(expressions: list[str]) -> str:
    """Evaluate labeled arithmetic expressions in exact Python `Decimal` (`ROUND_HALF_UP`).

    Rounding rationale: multi-addend sums are computed round-then-add (each addend
    quantized to 2 decimals before summation), mirroring how administrators post
    per-position rounded amounts on vouchers and ledgers; the exact 4-decimal raw
    value is reported alongside for reconciliation, so any drift between the two is
    visible rather than hidden. Single expressions quantize the final result only.

    Args:
        expressions: List of arithmetic expression strings (`Label = expr` or `expr`).

    Returns:
        Formatted multi-line string of deterministic `Decimal` results.
    """
    lines = ["[DETERMINISTIC DECIMAL (ROUND_HALF_UP) RESULTS]"]
    raw_symbols: dict[str, Decimal] = {}
    r2_symbols: dict[str, Decimal] = {}
    sym_registry: list[tuple[str, str]] = []

    for idx_expr, item in enumerate(expressions):
        label, expr = split_label_and_expr(item)
        clean_expr = normalize_calc_expression(expr, sym_registry)
        try:
            parsed = ast.parse(clean_expr, mode="eval")
            raw = eval_decimal_node(parsed, raw_symbols)
            r2 = round_currency(raw)
            r4 = raw.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            addends = collect_top_level_addends(parsed)

            has_sym_ref = "__sym_" in clean_expr or any(isinstance(n, ast.Name) for n in ast.walk(parsed))
            if has_sym_ref and isinstance(parsed.body, ast.BinOp) and isinstance(parsed.body.op, (ast.Add, ast.Sub)):
                r2 = round_currency(eval_decimal_node(parsed, r2_symbols))
            elif len(addends) > 1 and any(isinstance(a, ast.BinOp) and isinstance(a.op, (ast.Div, ast.Mult)) for a in addends):
                r2 = sum((round_currency(eval_decimal_node(a, raw_symbols)) for a in addends), Decimal("0.00"))

            display_expr = re.sub(
                r"\(__sym_\d+__\)",
                lambda m: next((lbl for lbl, tok in sym_registry if f"({tok})" == m.group(0)), m.group(0)),
                clean_expr,
            )
            lines.append(_fmt_line(label, display_expr, r2, str(r4)))
            if label:
                sym_tok = f"__sym_{idx_expr}__"
                norm_lbl = re.sub(r"\W+", "_", label.lower()).strip("_")
                raw_symbols[sym_tok], r2_symbols[sym_tok] = raw, r2
                if norm_lbl:
                    raw_symbols[norm_lbl], r2_symbols[norm_lbl] = raw, r2
                sym_registry.append((label, sym_tok))
        except (ValueError, ArithmeticError, SyntaxError) as e:
            lines.append(f"- `{item}` => ERROR: {e}")

    return "\n".join(lines)
