"""Safe AST `Decimal` (`ROUND_HALF_UP`) evaluator & expression normalizer (`ast_decimal_evaluator.py`).

Evaluates whitelisted arithmetic AST nodes in exact Python `Decimal` with profile-configured
exponent and digit-count DoS bounds, flattens top-level addends, and normalizes units/symbols.
"""

from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal

from domain_profile import PROFILE

_SAFE_DECIMAL_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _harness_cfg() -> dict:
    return PROFILE.get("harness_config", {})


def _build_unit_strip_re() -> re.Pattern:
    cfg = _harness_cfg()
    tokens = [*cfg.get("currencies", []), *cfg.get("units", [])]
    if not tokens:
        return re.compile(r"(?!x)x")
    return re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(tokens) + r")(?![A-Za-z0-9_])", re.IGNORECASE)


def eval_decimal_node(node: ast.AST, symbols: dict[str, Decimal] | None = None) -> Decimal:
    """Safely evaluate an AST arithmetic node using exact Python `Decimal`.

    Args:
        node: Parsed Python `ast.AST` expression node.
        symbols: Optional mapping of resolved variable/label tokens to `Decimal` values.

    Returns:
        Exact `Decimal` result of evaluating `node`.

    Raises:
        ValueError: If an unsupported AST node is encountered or an `ast.Pow` operation
            exceeds the profile-configured exponent cap or output digit limit.
    """
    cfg = _harness_cfg()
    max_exp = Decimal(str(cfg.get("max_ast_exponent", 1000)))
    max_digits = Decimal(str(cfg.get("max_precision_digits", 24)))

    if isinstance(node, ast.Expression):
        return eval_decimal_node(node.body, symbols)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.Name) and symbols and node.id in symbols:
        return symbols[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_DECIMAL_OPS:
        left_val = eval_decimal_node(node.left, symbols)
        right_val = eval_decimal_node(node.right, symbols)
        if isinstance(node.op, ast.Pow):
            if abs(right_val) > max_exp:
                raise ValueError(f"Exponent {right_val} exceeds maximum allowed limit ({max_exp})")
            if (
                abs(left_val) > Decimal("1")
                and abs(right_val) * Decimal(str(max(1, len(str(int(abs(left_val))))))) > max_digits
            ):
                raise ValueError(
                    f"Exponent {right_val} on base {left_val} exceeds {max_digits}-digit precision limit"
                )
        return _SAFE_DECIMAL_OPS[type(node.op)](left_val, right_val)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_DECIMAL_OPS:
        return _SAFE_DECIMAL_OPS[type(node.op)](eval_decimal_node(node.operand, symbols))
    raise ValueError(f"Unsupported arithmetic node: {type(node).__name__}")


def collect_top_level_addends(node: ast.AST) -> list[ast.AST]:
    """Flatten a chain of top-level `ast.Add` nodes into individual addend AST nodes.

    Args:
        node: Root `ast.AST` node.

    Returns:
        Ordered list of addend AST subnodes.
    """
    if isinstance(node, ast.Expression):
        return collect_top_level_addends(node.body)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return collect_top_level_addends(node.left) + collect_top_level_addends(node.right)
    return [node]


def normalize_calc_expression(expr: str, sym_registry: list[tuple[str, str]]) -> str:
    """Normalize raw arithmetic string into valid Python syntax and resolve prior line labels.

    Args:
        expr: Raw arithmetic expression string (may contain Unicode operators, units, or labels).
        sym_registry: Ordered list of `(raw_label, sym_token)` pairs from earlier lines.

    Returns:
        Normalized Python arithmetic expression string ready for `ast.parse(..., mode='eval')`.
    """
    s = (
        expr.strip()
        .replace("×", "*")
        .replace("⋅", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("^", "**")
    )
    for raw_lbl, sym_tok in sorted(sym_registry, key=lambda x: len(x[0]), reverse=True):
        if len(raw_lbl) >= 2:
            s = re.sub(rf"\(\s*{re.escape(raw_lbl)}\s*\)", f"({sym_tok})", s, flags=re.IGNORECASE)
            s = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(raw_lbl)}(?![A-Za-z0-9_])",
                f"({sym_tok})",
                s,
                flags=re.IGNORECASE,
            )

    s = re.sub(r"(?<=\d)[,_](?=\d)", "", s)
    s = _build_unit_strip_re().sub("", s)
    if "%" in s:
        if "*" in s or "/" in s:
            s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100)", s)
        else:
            s = s.replace("%", "")
    return s.strip()
