"""
Verification Engine for Tier 2: Agent Reasoning & Symbolic Calculation Benchmark.

Provides dual-fidelity numerical verification, citation grounding analysis,
streaming protocol marker scanning, and zero-LaTeX enforcement.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any


_STATUTORY_CITATION_NUM_RE = re.compile(
    r"\b(?:Section|Sec\.|Article|Art\.|Table|Point|Paragraph|Chapter|Circular|CSSF)\s+\d+(?:[./]\d+)*",
    re.IGNORECASE,
)
_ISO_DATE_NUM_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def extract_decimal_numbers(text_or_obj: Any) -> list[Decimal]:
    """Extract all valid decimal numbers from a string or object.

    Args:
        text_or_obj: Input text, json, or object to scan.

    Returns:
        List of Decimal numbers parsed with commas removed.
    """
    if not text_or_obj:
        return []
    s = text_or_obj if isinstance(text_or_obj, str) else json.dumps(text_or_obj)
    s = s.replace(",", "")
    nums: list[Decimal] = []
    for token in re.findall(r"[-+]?\d*\.?\d+", s):
        try:
            nums.append(Decimal(token))
        except Exception:
            continue
    return nums


def extract_prose_decimal_numbers(text: str) -> list[Decimal]:
    """Extract decimal numbers from final prose after stripping statutory section/table IDs and ISO dates."""
    if not text:
        return []
    cleaned = _STATUTORY_CITATION_NUM_RE.sub(" ", text)
    cleaned = _ISO_DATE_NUM_RE.sub(" ", cleaned)
    return extract_decimal_numbers(cleaned)


def extract_tool_output_numbers(calc_tool_outputs: list[str]) -> list[Decimal]:
    """Extract computed decimal numbers from tool outputs without mistaking echoed LHS currency/share operands."""
    nums: list[Decimal] = []
    for out in calc_tool_outputs or []:
        for line in str(out).splitlines():
            if "=>" in line:
                lhs, rhs = line.split("=>", 1)
                rhs_clean = re.sub(r"\(exact 4-dec:[^)]*\)", "", rhs)
                nums.extend(extract_decimal_numbers(rhs_clean))
                for n in extract_decimal_numbers(lhs):
                    if Decimal("0") < abs(n) <= Decimal("100"):
                        nums.append(n)
            else:
                nums.extend(extract_decimal_numbers(line))
    return nums


def _audit_single_candidate(
    gt_dec: Decimal,
    tol: Decimal,
    prompt_nums: list[Decimal],
    final_nums: list[Decimal],
    tool_nums: list[Decimal],
) -> dict[str, Any]:
    """Audit a single numeric ground-truth value against tool outputs and final prose."""
    tool_diffs = [abs(n - gt_dec) for n in tool_nums]
    min_tool_diff = min(tool_diffs) if tool_diffs else None
    in_tool = (min_tool_diff is not None) and (min_tool_diff <= tol)

    final_diffs = [abs(n - gt_dec) for n in final_nums]
    min_final_diff = min(final_diffs) if final_diffs else None
    in_final = (min_final_diff is not None) and (min_final_diff <= tol)

    in_prompt = any(abs(n - gt_dec) <= tol for n in prompt_nums)

    if in_tool and in_final:
        status = "VERIFIED_DUAL_FIDELITY"
        matched = True
    elif in_tool and not in_final:
        status = "TRANSCRIPTION_ERROR"
        matched = False
    elif not in_tool and in_final:
        status = "PROMPT_ECHO_NO_TOOL" if in_prompt else "MENTAL_MATH_NO_TOOL"
        matched = False
    else:
        status = "MISSING_IN_BOTH"
        matched = False

    return {
        "expected": float(gt_dec),
        "in_tool": in_tool,
        "in_final": in_final,
        "in_prompt": in_prompt,
        "matched": matched,
        "status": status,
        "tool_diff": float(min_tool_diff) if min_tool_diff is not None else None,
        "final_diff": float(min_final_diff) if min_final_diff is not None else None,
    }


def verify_dual_numeric_fidelity(
    gt_numbers: list[float | int | dict[str, Any] | list[float]],
    prompt: str,
    final_text: str,
    calc_tool_outputs: list[str],
    tolerance: float = 0.01,
) -> tuple[bool, list[dict[str, Any]]]:
    """Verify dual provenance: (1) machine execution in ToolMessage AND (2) reported in final answer.

    Supports both scalar numbers and `{"any_of": [v1, v2, ...]}` (or list) alternatives
    for scenarios with multiple valid accounting bases (e.g., reported vs. replicated NAV bps,
    or net-flow vs. gross-subscription dilution).

    Args:
        gt_numbers: List of expected ground truth figures or `{"any_of": [...]}` specs.
        prompt: Original user question prompt (to detect echoed numbers).
        final_text: Final synthesized response presented to user.
        calc_tool_outputs: Raw content strings from calculate_decimal and run_nav_audit ToolMessages.
        tolerance: Numerical tolerance (default 0.01 EUR for penny rounding).

    Returns:
        Tuple of (all_numbers_passed: bool, detailed_audit_records: list[dict]).
    """
    if not gt_numbers:
        return True, []

    tol = Decimal(str(tolerance))
    prompt_nums = extract_decimal_numbers(prompt)
    final_nums = extract_prose_decimal_numbers(final_text)
    tool_nums = extract_tool_output_numbers(calc_tool_outputs)

    all_ok = True
    audit_records: list[dict[str, Any]] = []

    for gt_entry in gt_numbers:
        if isinstance(gt_entry, dict) and "any_of" in gt_entry:
            candidates = list(gt_entry["any_of"])
        elif isinstance(gt_entry, (list, tuple)):
            candidates = list(gt_entry)
        else:
            candidates = [gt_entry]

        cand_records = [
            _audit_single_candidate(Decimal(str(c)), tol, prompt_nums, final_nums, tool_nums)
            for c in candidates
        ]
        best_record = next(
            (r for r in cand_records if r["matched"]),
            next((r for r in cand_records if r["in_tool"] or r["in_final"]), cand_records[0]),
        )
        if len(candidates) > 1:
            best_record["any_of"] = [float(Decimal(str(c))) for c in candidates]

        if not best_record["matched"]:
            all_ok = False
        audit_records.append(best_record)

    return all_ok, audit_records


from evaluation.agent_rag_grounding_verifier import (
    extract_retrieved_documents,
    match_single_citation,
    normalize_text_for_matching,
    verify_citations,
    verify_retrieval_fidelity,
)


def check_protocol_markers(text: str) -> dict[str, bool]:
    """Check 5 streaming protocol markers and zero-LaTeX compliance.

    Args:
        text: Final synthesized response.

    Returns:
        Dict of marker statuses and overall protocol_passed flag.
    """
    has_reasoning = "[REASONING]" in text
    has_conclusion_marker = "[CONCLUSION]" in text
    has_conclusion_heading = "### CONCLUSION" in text or "### Conclusion:" in text
    has_detailed_heading = any(h in text for h in ("### DETAILED ANALYSIS", "### Detailed Analysis:", "### Detailed analysis:"))
    has_sources = any(s in text for s in ("**SOURCES**", "**Sources:**", "**INTERNET SOURCES**"))
    zero_latex = not any(kw in text for kw in ("\\times", "\\frac", "\\text{"))
    is_tier1_short_circuit = "⚠️ **Out of Scope:**" in text or "⚠️ **Not found (Out of Scope):**" in text

    return {
        "has_reasoning": has_reasoning,
        "has_conclusion_marker": has_conclusion_marker,
        "has_conclusion_heading": has_conclusion_heading,
        "has_detailed_heading": has_detailed_heading,
        "has_sources": has_sources,
        "zero_latex": zero_latex,
        "protocol_passed": (
            all([has_reasoning, has_conclusion_marker, has_sources])
            if is_tier1_short_circuit
            else all([has_reasoning, has_conclusion_marker, has_conclusion_heading, has_detailed_heading, has_sources])
        ),
    }


def evaluate_response(
    scenario: dict[str, Any],
    text: str,
    tool_calls: list[dict[str, Any]],
    calc_tool_outputs: list[str],
    search_tool_outputs: list[str] | None = None,
    elapsed: float = 0.0,
) -> dict[str, Any]:
    """Audit single scenario execution against ground truth rubric.

    Args:
        scenario: Benchmark scenario dictionary.
        text: Agent final prose text.
        tool_calls: List of tool calls captured from execution graph.
        calc_tool_outputs: Captured calculation ToolMessage content strings.
        search_tool_outputs: Captured search_documents ToolMessage content strings.
        elapsed: Elapsed latency in seconds.

    Returns:
        Dictionary of verification verdicts and diagnostic audit trail.
    """
    called_tool_names = [tc.get("name") for tc in tool_calls]
    category = scenario.get("category", "calculation")

    # 1. Tool Delegation Check (Generic without hardcoded scenario IDs)
    req_tools = scenario.get("required_tools", [])
    is_abstention = scenario.get("is_abstention", False) or (category == "rag_grounding" and len(req_tools) == 0)

    if category == "calculation":
        tool_delegation_ok = any(t in ("calculate_decimal", "run_nav_audit") for t in called_tool_names)
    elif is_abstention:
        tool_delegation_ok = True
    else:
        if req_tools:
            tool_delegation_ok = any(t in called_tool_names for t in req_tools)
        else:
            tool_delegation_ok = "search_documents" in called_tool_names or len(called_tool_names) > 0

    # 2. Dual-Fidelity Number Check (Tool Provenance + Final Answer Reporting)
    tol = scenario.get("tolerance") or 0.01
    if category == "calculation":
        nums_ok, num_audit = verify_dual_numeric_fidelity(
            gt_numbers=scenario.get("ground_truth_numbers", []),
            prompt=scenario.get("prompt", ""),
            final_text=text,
            calc_tool_outputs=calc_tool_outputs,
            tolerance=tol,
        )
    else:
        # RAG Grounding scenarios skip number verification; no number_audit produced
        nums_ok, num_audit = True, None

    # 3. Citation Check
    citations_ok = verify_citations(scenario.get("expected_citations", []), text)

    # 4. RAG Retrieval Telemetry Check (Authentic Document Verification)
    expected_docs = scenario.get("expected_source_documents", [])
    retrieval_ok, matched_docs, missing_docs = verify_retrieval_fidelity(expected_docs, search_tool_outputs or [])

    # 5. Protocol Checks
    proto = check_protocol_markers(text)

    # Failure reasons explanation
    failure_reasons = []
    if category == "calculation":
        if not tool_delegation_ok:
            failure_reasons.append("Symbolic calculation tool not called (Zero Mental Math breach).")
        for item in (num_audit or []):
            if not item["matched"]:
                failure_reasons.append(
                    f"Expected {item['expected']} failed: {item.get('status')} (ToolDiff={item['tool_diff']}, FinalDiff={item['final_diff']})"
                )
    else:
        if not tool_delegation_ok:
            failure_reasons.append("Required RAG retrieval tool not invoked.")
        if not citations_ok:
            failure_reasons.append(f"Missing mandatory statutory citations: {scenario.get('expected_citations')}")
        if not retrieval_ok:
            failure_reasons.append(f"Corpus retrieval miss: missing required documents {missing_docs}")
    if not proto["protocol_passed"]:
        failure_reasons.append("Streaming protocol markers incomplete.")
    if not proto["zero_latex"]:
        failure_reasons.append("LaTeX formatting detected in final prose.")

    # Final Verdict
    passed = (
        (tool_delegation_ok and nums_ok if category == "calculation" else tool_delegation_ok and citations_ok and retrieval_ok)
        and proto["protocol_passed"]
        and proto["zero_latex"]
    )

    return {
        "id": scenario["id"],
        "title": scenario["title"],
        "category": category,
        "elapsed_sec": round(elapsed, 2),
        "prompt": scenario.get("prompt", ""),
        "final_text": text,
        "ground_truth_numbers": scenario.get("ground_truth_numbers", []),
        "tolerance": tol,
        "expected_citations": scenario.get("expected_citations", []),
        "expected_source_documents": expected_docs,
        "tool_delegation_ok": tool_delegation_ok,
        "tools_called": called_tool_names,
        "tool_calls": tool_calls,
        "calc_tool_outputs": calc_tool_outputs or [],
        "search_tool_outputs": search_tool_outputs or [],
        "retrieved_documents": extract_retrieved_documents(search_tool_outputs or []),
        "numbers_ok": nums_ok,
        "number_audit": num_audit,
        "citations_ok": citations_ok,
        "retrieval_ok": retrieval_ok,
        "matched_documents": matched_docs,
        "missing_documents": missing_docs,
        "protocol": proto,
        "passed": passed,
        "failure_reasons": failure_reasons,
    }
