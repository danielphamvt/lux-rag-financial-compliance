"""Multi-turn calculation continuity & context-compression telemetry (`multi_turn_calc_continuity.py`).

Unifies `=` and `:` label splitting across calculation calls, merges prior leg expressions
into follow-up `calculate_decimal` calls, and builds compact telemetry stubs across context compression.
"""

from __future__ import annotations

import re
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

_SUMMARY_KWS = ("total", "aggregate", "combined", "sum", "subtotal", "basket")
_DELTA_KWS = (
    "diff",
    "variance",
    "overstatement",
    "understatement",
    "excess",
    "breach",
    "headroom",
    "margin",
    "buffer",
    "remaining",
    "shortfall",
)


def split_label_and_expr(item: str) -> tuple[str, str]:
    """Split an optional `Label = expr` or `Label: expr` prefix from an arithmetic expression.

    Args:
        item: Raw expression string passed to `calculate_decimal`.

    Returns:
        Tuple of `(cleaned_label, cleaned_expression)`. When unlabeled, `cleaned_label` is `""`.
    """
    if "=" in item and not any(op in item for op in ("<=", ">=", "!=", "==")):
        label, _, expr = item.partition("=")
        if expr.strip():
            return label.strip(), expr.strip()
    elif ":" in item:
        label, _, expr = item.partition(":")
        if expr.strip():
            return label.strip(), expr.strip()
    return "", item.strip()


def _extract_raw_exprs(items: list | str | None) -> list[str]:
    if not items:
        return []
    return [items] if isinstance(items, str) else list(items)


def _collect_leg_expressions(agent_messages: list | None, ledger: list | None = None) -> list[str]:
    """Collect deduplicated non-summary calculation leg expressions from `agent_messages` and `ledger`."""
    raw_expr_lists: list[list[str]] = []
    for msg in agent_messages or []:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.get("name") == "calculate_decimal":
                    raw_expr_lists.append(_extract_raw_exprs((tc.get("args") or {}).get("expressions")))
    for rec in ledger or []:
        if getattr(rec, "tool_name", "") == "calculate_decimal":
            raw_expr_lists.append(_extract_raw_exprs((getattr(rec, "args", None) or {}).get("expressions")))

    prior_legs: list[str] = []
    seen_labels: set[str] = set()
    for exprs in raw_expr_lists:
        for item in exprs:
            lbl, rhs = split_label_and_expr(item)
            lbl_clean = lbl.lower()
            if not rhs or any(w in lbl_clean for w in _SUMMARY_KWS + _DELTA_KWS):
                continue
            key = lbl_clean or item.strip()
            if key not in seen_labels:
                seen_labels.add(key)
                prior_legs.append(item)
    return prior_legs


def merge_prior_calc_legs(
    first_arg: list,
    second_arg: list,
    ledger: list | None = None,
) -> list[str]:
    """Prepend earlier leg expressions from the current turn into follow-up `calculate_decimal` calls.

    Prior-leg evidence is the union of `agent_messages` tool calls and the durable
    tool-execution ledger, so legs survive context compression.

    Accepts either `(agent_messages, tool_calls)` (mutating `tool_calls` in-place) or
    `(expressions_list, agent_messages)` (returning the merged expression list).

    Args:
        first_arg: Either `agent_messages` or a list of expression strings.
        second_arg: Either `tool_calls` dicts or `agent_messages`.
        ledger: Optional durable `ToolExecutionRecord` list from graph state.

    Returns:
        Merged list of expression strings with prior legs prepended and deduplicated.
    """
    if not first_arg and not second_arg and not ledger:
        return []
    if not first_arg or not second_arg:
        return list(first_arg) if first_arg and isinstance(first_arg[0], str) else []

    if isinstance(first_arg[0], str):
        direct_exprs: list[str] | None = list(first_arg)
        agent_messages = second_arg
        tool_calls = [{"name": "calculate_decimal", "args": {"expressions": direct_exprs}}]
    else:
        direct_exprs = None
        agent_messages = first_arg
        tool_calls = second_arg

    prior_legs = _collect_leg_expressions(agent_messages, ledger)
    merged_out: list[str] = direct_exprs or []
    for tc in tool_calls:
        if tc.get("name") == "calculate_decimal" and isinstance(tc.get("args"), dict):
            curr = _extract_raw_exprs(tc["args"].get("expressions"))
            curr_keys = {
                (split_label_and_expr(item)[0].lower() or item.strip())
                for item in curr
            }
            to_prepend = [
                leg
                for leg in prior_legs
                if (split_label_and_expr(leg)[0].lower() or leg.strip()) not in curr_keys
            ]
            merged_out = (to_prepend + curr) if to_prepend else curr
            tc["args"]["expressions"] = merged_out
    return merged_out


def build_compressed_record_messages(ledger: list) -> list:
    """Build protocol-valid compressed-history stubs from the durable tool-execution ledger.

    Args:
        ledger: `ToolExecutionRecord` list from graph state (`tool_execution_log`).

    Returns:
        Synthetic `AIMessage(tool_calls)` + named `ToolMessage` pairs labeled as
        compressed history (id namespace `digest_*`), or an empty list when the
        ledger holds no usable records.
    """
    records = [r for r in (ledger or []) if getattr(r, "tool_name", "")]
    calc_exprs: list[str] = []
    audit_args: list[dict] = []
    math_lines: list[str] = []
    corpus_hdrs: list[str] = []
    low_hdrs: list[str] = []
    for r in records:
        out_s = str(getattr(r, "output", "") or "")
        if r.tool_name == "calculate_decimal":
            for e in _extract_raw_exprs((r.args or {}).get("expressions")):
                if e not in calc_exprs:
                    calc_exprs.append(e)
        elif r.tool_name == "run_nav_audit":
            audit_args.append(dict(r.args or {}))
        if r.tool_name in ("calculate_decimal", "run_nav_audit"):
            math_lines.extend(ln for ln in out_s.splitlines() if "**" in ln and ln not in math_lines)
        elif r.tool_name == "search_documents":
            target = low_hdrs if "[LOW_RELEVANCE_WARNING:" in out_s else corpus_hdrs
            target.extend(h for h in re.findall(r"^(?:Document ID|File Name):\s*[^\n]+", out_s, re.M) if h not in target)

    label = f"[COMPRESSED TOOL HISTORY — {len(records)} tool executions]"
    tcs: list[dict] = []
    tms: list[ToolMessage] = []

    def _add_stub(name: str, args: dict, body: str) -> None:
        tc_id = f"digest_{name}_{uuid4().hex[:8]}"
        tcs.append({"name": name, "args": args, "id": tc_id, "type": "tool_call"})
        tms.append(ToolMessage(content=f"{label}\n{body}", tool_call_id=tc_id, name=name, id=f"digest_tm_{uuid4().hex[:8]}"))

    if calc_exprs or (math_lines and not audit_args):
        _add_stub("calculate_decimal", {"expressions": calc_exprs}, "\n".join(math_lines) or "[COMPRESSED DECIMAL STEPS]")
    elif audit_args:
        _add_stub("run_nav_audit", audit_args[-1], "\n".join(math_lines) or "[COMPRESSED NAV AUDIT STEPS]")
    if corpus_hdrs or low_hdrs:
        hdr_body = "\n".join(corpus_hdrs) if corpus_hdrs else "[LOW_RELEVANCE_WARNING: compressed]\n" + "\n".join(low_hdrs)
        _add_stub("search_documents", {"query": "compressed_history"}, hdr_body)

    return [AIMessage(content="", tool_calls=tcs, id=f"digest_ai_{uuid4().hex[:8]}"), *tms] if tcs else []
