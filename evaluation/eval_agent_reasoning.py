"""
Automated Evaluation Suite for Tier 2: End-to-End Agent Reasoning & Calculation.

Verifies:
- Dataset integrity (40 scenarios: 30 Calculation + 10 RAG Grounding).
- Protocol checklist & zero-LaTeX enforcement.
- Numeric verification logic with ±0.01 EUR penny rounding tolerance.
- Mock execution flow and benchmark metric aggregation.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

# Strictly enforce isolated test environment (.env.test)
from evaluation.eval_common import ensure_runtime

ensure_runtime()

import pytest

from evaluation.agent_audit_reporter import save_markdown_audit_report
from evaluation.agent_reasoning_verifier import (
    check_protocol_markers,
    evaluate_response,
    verify_citations,
    verify_dual_numeric_fidelity,
    verify_retrieval_fidelity,
)
from evaluation.evaluate_agent_calculation import (
    DATASET_PATH,
    load_dataset,
    run_benchmark,
)


def test_tier2_dataset_structure_and_distribution() -> None:
    """Verify that the 40-scenario benchmark dataset is well-formed and complete."""
    scenarios = load_dataset(DATASET_PATH)
    assert len(scenarios) == 40, f"Expected 40 scenarios, found {len(scenarios)}"

    calc_scenarios = [s for s in scenarios if s.get("category") == "calculation"]
    rag_scenarios = [s for s in scenarios if s.get("category") == "rag_grounding"]

    assert len(calc_scenarios) == 30, f"Expected 30 calculation scenarios, got {len(calc_scenarios)}"
    assert len(rag_scenarios) == 10, f"Expected 10 RAG scenarios, got {len(rag_scenarios)}"

    required_keys = {"id", "category", "level", "title", "prompt", "required_tools", "expected_verdict", "rubric"}
    ids_seen = set()

    for sc in scenarios:
        missing = required_keys - set(sc.keys())
        assert not missing, f"Scenario {sc.get('id')} missing keys: {missing}"
        assert sc["id"] not in ids_seen, f"Duplicate scenario ID: {sc['id']}"
        ids_seen.add(sc["id"])

        if sc["category"] == "calculation":
            assert len(sc.get("ground_truth_numbers", [])) > 0, f"Calc scenario {sc['id']} has no GT numbers"
            assert any(t in ("calculate_decimal", "run_nav_audit") for t in sc["required_tools"]), (
                f"Calc scenario {sc['id']} must require calculate_decimal or run_nav_audit"
            )
        elif sc["category"] == "rag_grounding":
            assert len(sc.get("ground_truth_numbers", [])) == 0, f"RAG scenario {sc['id']} must not have GT numbers"
            assert len(sc.get("expected_citations", [])) > 0, f"RAG scenario {sc['id']} has no expected citations"


def test_tier2_protocol_checker() -> None:
    """Verify streaming protocol markers and zero-LaTeX detection."""
    valid_text = (
        "[REASONING]\nStep 1: Check Section 9.1.\n[CONCLUSION]\n"
        "### CONCLUSION\nTotal fee is 110,958.90 EUR.\n"
        "### DETAILED ANALYSIS\nFormula: 75,000,000 * 0.012 * 45 / 365.\n"
        "---\n**SOURCES**\n- lux_alpha_ucits_prospectus.md"
    )
    res_valid = check_protocol_markers(valid_text)
    assert res_valid["protocol_passed"] is True
    assert res_valid["zero_latex"] is True

    # Missing heading
    invalid_text = "[REASONING]\nTest\n[CONCLUSION]\nNo headings here\n---\n**SOURCES**"
    res_invalid = check_protocol_markers(invalid_text)
    assert res_invalid["protocol_passed"] is False

    # Leaked LaTeX
    latex_text = valid_text + "\nCalculation: \\times 100 \\frac{1}{2}"
    res_latex = check_protocol_markers(latex_text)
    assert res_latex["zero_latex"] is False


def test_tier2_number_verification_tolerance() -> None:
    """Verify numeric extraction with ±0.01 EUR penny rounding tolerance and Dual-Verification."""
    gt_numbers = [110958.90, 626712.32]
    tool_exact = ["- Result 1: `expr` => **110,958.90**", "- Result 2: `expr` => **626,712.32**"]
    text_exact = "Result 1 is 110958.90 EUR and Result 2 is 626712.32 EUR"

    # 1. Exact match in both ToolMessage and Final Answer -> PASS
    ok_exact, details_exact = verify_dual_numeric_fidelity(
        gt_numbers=gt_numbers, prompt="", final_text=text_exact, calc_tool_outputs=tool_exact, tolerance=0.01
    )
    assert ok_exact is True
    assert all(d["status"] == "VERIFIED_DUAL_FIDELITY" for d in details_exact)

    # 2. Transcription Error: Tool computed correctly, but final text made a typo -> FAIL
    text_transcription_err = "Result 1 is 110958.90 EUR and Result 2 is 626000.00 EUR"
    ok_trans, details_trans = verify_dual_numeric_fidelity(
        gt_numbers=gt_numbers, prompt="", final_text=text_transcription_err, calc_tool_outputs=tool_exact, tolerance=0.01
    )
    assert ok_trans is False
    assert details_trans[0]["status"] == "VERIFIED_DUAL_FIDELITY"
    assert details_trans[1]["status"] == "TRANSCRIPTION_ERROR"

    # 3. Prompt Echo: Number exists in prompt and final answer, but NEVER in ToolMessage -> FAIL
    prompt_with_num = "Please verify why NAV was 110958.90 EUR yesterday"
    text_echo = "NAV was 110958.90 EUR yesterday"
    ok_echo, details_echo = verify_dual_numeric_fidelity(
        gt_numbers=[110958.90], prompt=prompt_with_num, final_text=text_echo, calc_tool_outputs=[], tolerance=0.01
    )
    assert ok_echo is False
    assert details_echo[0]["status"] == "PROMPT_ECHO_NO_TOOL"

    # 4. Mental Math: Number in final answer but neither in ToolMessage nor in prompt -> FAIL
    ok_mental, details_mental = verify_dual_numeric_fidelity(
        gt_numbers=[110958.90], prompt="", final_text=text_echo, calc_tool_outputs=[], tolerance=0.01
    )
    assert ok_mental is False
    assert details_mental[0]["status"] == "MENTAL_MATH_NO_TOOL"

    # 5. Any-of alternatives (dual-denominator bps or dual-basis dilution): matching either basis passes
    ok_any_net, details_any_net = verify_dual_numeric_fidelity(
        gt_numbers=[3.6, 251.0, {"any_of": [35856.57, 47808.76]}],
        prompt="",
        final_text="Net flow is 3.60%, swung NAV is 251.00 EUR, and net dilution adjustment is 35,856.57 EUR.",
        calc_tool_outputs=["- Flow: 3.60\n- Swung NAV: 251.00\n- Dilution: 35,856.57"],
        tolerance=0.05,
    )
    assert ok_any_net is True
    assert details_any_net[2]["matched"] is True
    assert details_any_net[2]["any_of"] == [35856.57, 47808.76]

    ok_any_gross, _ = verify_dual_numeric_fidelity(
        gt_numbers=[3.6, 251.0, {"any_of": [35856.57, 47808.76]}],
        prompt="",
        final_text="Net flow is 3.60%, swung NAV is 251.00 EUR, and gross dilution adjustment is 47,808.76 EUR.",
        calc_tool_outputs=["- Flow: 3.60\n- Swung NAV: 251.00\n- Dilution: 47,808.76"],
        tolerance=0.05,
    )
    assert ok_any_gross is True

    # 6. Anti-spoofing guard: LHS input operand (100,000 shares) before '=>' and statutory section IDs ('Section 5.5') do not falsely pass
    ok_lhs_spoof, details_lhs_spoof = verify_dual_numeric_fidelity(
        gt_numbers=[100000.0, 5.5],
        prompt="A fund sold 100,000 shares.",
        final_text="The fund sold 100,000 shares under Section 5.5.",
        calc_tool_outputs=["- Realized loss: `100000 * (50.00 - 46.00)` => **400,000.00** (exact 4-dec: `400000.0000`)"],
        tolerance=0.01,
    )
    assert ok_lhs_spoof is False
    assert details_lhs_spoof[0]["in_tool"] is False
    assert details_lhs_spoof[1]["in_final"] is False


def test_tier2_rag_grounding_skips_number_verification() -> None:
    """Verify that all 10 RAG Grounding scenarios skip number verification and have no number_audit."""
    scenarios = load_dataset(DATASET_PATH)
    rag_scenarios = [s for s in scenarios if s.get("category") == "rag_grounding"]
    assert len(rag_scenarios) == 10, f"Expected 10 RAG scenarios, got {len(rag_scenarios)}"

    for sc in rag_scenarios:
        # 1. Dataset invariant: no ground_truth_numbers assigned
        assert sc.get("ground_truth_numbers", []) == [], f"{sc['id']} must have empty ground_truth_numbers"

        # 2. Mock evaluation execution
        cits = sc.get("expected_citations", ["CSSF 24/856"])
        exp_docs = sc.get("expected_source_documents", ["CSSF24_856eng.pdf"])
        mock_text = (
            f"[REASONING]\nAuditing legal requirements for {sc['id']}.\n[CONCLUSION]\n"
            f"### CONCLUSION\nRequirement verified per statutory provisions.\n"
            f"### DETAILED ANALYSIS\nGoverned by {', '.join(cits)}.\n"
            f"---\n**SOURCES**\n- {exp_docs[0] if exp_docs else 'corpus.md'}"
        )
        mock_search = [
            f"Parent ID: 1\nDocument ID: {d}\nFile Name: {d}\nSection / Page: Sec 1\nContent: Legal text"
            for d in exp_docs
        ]
        tool_calls = (
            []
            if (sc.get("is_abstention") or len(sc.get("required_tools", [])) == 0)
            else [{"name": "search_documents", "args": {"query": sc["prompt"][:40]}}]
        )

        res = evaluate_response(
            sc,
            text=mock_text,
            tool_calls=tool_calls,
            calc_tool_outputs=[],
            search_tool_outputs=mock_search,
        )

        # 3. Numeric verification must be completely bypassed (numbers_ok=True, number_audit=None)
        assert res["numbers_ok"] is True, f"{sc['id']} numbers_ok should be True"
        assert res["number_audit"] is None, f"{sc['id']} number_audit should be None"

        # 4. Formatted Num Match string must be '-' (not displaying any numeric match ratio)
        num_m = "-" if (sc.get("category") != "calculation" or not res.get("number_audit")) else f"{sum(1 for d in res['number_audit'] if d['matched'])}/{len(res['number_audit'])}"
        assert num_m == "-", f"{sc['id']} Num Match display must be '-'"


def test_tier2_citation_verification() -> None:
    """Verify statutory citation matching."""
    expected = ["Section 9.1", "Actual/365", "Lux Alpha"]
    text_good = "According to Section 9.1 and Actual/365 of the Lux Alpha Prospectus..."
    assert verify_citations(expected, text_good) is True

    text_poor = "According to general accounting standards..."
    assert verify_citations(expected, text_poor) is False


def test_tier2_flexible_citation_matching() -> None:
    """Verify robust matching of abbreviations, number delimiters, basis points, and abstention equivalents."""
    # 1. Abbreviations: Art. 52 vs Article 52, Sec. 9.1 vs Section 9.1
    assert verify_citations(["Article 52", "Section 9.1"], "Governed by Art. 52 and Sec. 9.1.") is True

    # 2. Number delimiters: 50,000 vs 50000
    assert verify_citations(["Table 10", "50,000", "5,000"], "Threshold of 50000 EUR or 5000 EUR per Table 10.") is True

    # 3. Basis points vs percent vs text
    assert verify_citations(["Table 1", "50 bps"], "Threshold is 50 basis points per Table 1.") is True
    assert verify_citations(["Table 1", "50 bps"], "Threshold is 0.50% per Table 1.") is True

    # 4. Semantic abstention / out-of-scope equivalents
    assert verify_citations(["out of scope", "not governed"], "This issue is outside the scope of Luxembourg law and not covered by CSSF.") is True


def test_tier2_retrieval_fidelity_multi_doc_tolerance() -> None:
    """Verify multi-document retrieval fidelity logic and graceful partial coverage."""
    # 0 docs: Abstention always passes
    hit_0, matched_0, missing_0 = verify_retrieval_fidelity([], [])
    assert hit_0 is True

    # 1 doc: Strict 100% required
    doc_out = ["Parent ID: 1\nDocument ID: CSSF24_856eng.pdf\nFile Name: CSSF24_856eng.pdf\nContent: txt"]
    hit_1_ok, _, _ = verify_retrieval_fidelity(["CSSF24_856eng.pdf"], doc_out)
    assert hit_1_ok is True
    hit_1_fail, _, _ = verify_retrieval_fidelity(["CSSF24_856eng.pdf"], [])
    assert hit_1_fail is False

    # 2 docs: At least 1 primary doc required (>= 50%)
    hit_2_ok, m2, miss2 = verify_retrieval_fidelity(["CSSF24_856eng.pdf", "cssf02_77eng.pdf"], doc_out)
    assert hit_2_ok is True
    assert len(m2) == 1 and len(miss2) == 1

    # 3 docs: At least 2 of 3 (>= 60%) required
    doc_out_2 = [
        "Parent ID: 1\nDocument ID: UCITS_Directive_2009_65_EC.md\nFile Name: UCITS_Directive_2009_65_EC.md\nContent: txt",
        "Parent ID: 2\nDocument ID: CSSF24_856eng.pdf\nFile Name: CSSF24_856eng.pdf\nContent: txt",
    ]
    three_docs = ["UCITS_Directive_2009_65_EC.md", "cssf02_77eng.pdf", "CSSF24_856eng.pdf"]
    hit_3_ok, m3, miss3 = verify_retrieval_fidelity(three_docs, doc_out_2)
    assert hit_3_ok is True  # 2 out of 3 = 66.7% >= 60%
    assert len(m3) == 2 and len(miss3) == 1

    # 1 out of 3 fails (33.3% < 60%)
    hit_3_fail, _, _ = verify_retrieval_fidelity(three_docs, doc_out)
    assert hit_3_fail is False


def test_tier2_generic_abstention_delegation() -> None:
    """Verify that any scenario flagged with is_abstention or required_tools=[] passes delegation without hardcoded ID."""
    custom_sc = {
        "id": "CUSTOM_ABSTAIN_99",
        "category": "rag_grounding",
        "level": 4,
        "title": "Custom Abstention Test",
        "prompt": "Is SEC Rule 10b-5 applicable?",
        "required_tools": [],
        "is_abstention": True,
        "ground_truth_numbers": [],
        "expected_citations": ["not applicable", "out of scope"],
        "expected_source_documents": [],
    }
    valid_text = (
        "[REASONING]\nChecking regulatory scope.\n[CONCLUSION]\n"
        "### CONCLUSION\nNot applicable in Luxembourg.\n"
        "### DETAILED ANALYSIS\nThis rule is outside the scope of CSSF.\n"
        "---\n**SOURCES**\n- None"
    )
    res = evaluate_response(custom_sc, text=valid_text, tool_calls=[], calc_tool_outputs=[])
    assert res["tool_delegation_ok"] is True
    assert res["citations_ok"] is True
    assert res["passed"] is True


def test_tier2_mock_benchmark_execution(tmp_path: Path) -> None:
    """Verify mock end-to-end benchmark execution and metric rollup."""
    scenarios = load_dataset(DATASET_PATH)[:6]  # Test first 6 scenarios
    out_file = tmp_path / "mock_agent_benchmark.json"

    summary = run_benchmark(scenarios, use_mock=True, output_path=out_file)
    assert summary["total_scenarios"] == 6
    assert summary["passed_scenarios"] == 6
    assert summary["pass_rate_pct"] == 100.0
    assert summary["calculation_delegation_rate_pct"] == 100.0
    assert summary["numerical_accuracy_rate_pct"] == 100.0
    assert out_file.is_file()


def test_tier2_concurrent_execution(tmp_path: Path) -> None:
    """Verify that concurrent execution (concurrency=3) preserves indexing and metric fidelity."""
    scenarios = load_dataset(DATASET_PATH)[:9]  # 9 scenarios tested with 3 workers
    out_file = tmp_path / "mock_concurrent_benchmark.json"

    summary = run_benchmark(scenarios, use_mock=True, output_path=out_file, concurrency=3)
    assert summary["total_scenarios"] == 9
    assert summary["passed_scenarios"] == 9
    assert summary["concurrency"] == 3
    # Check that original order was preserved
    res_ids = [r["id"] for r in summary["results"]]
    exp_ids = [s["id"] for s in scenarios]
    assert res_ids == exp_ids
    assert out_file.is_file()


def test_save_markdown_audit_report_with_rag_and_calc_failures(tmp_path: Path) -> None:
    """Verify that save_markdown_audit_report handles both calc failures and RAG failures (where number_audit is None)."""
    fake_summary = {
        "timestamp": "2026-10-26T13:00:00",
        "run_timestamp": "2026-10-26-13-00",
        "total_scenarios": 2,
        "passed_scenarios": 0,
        "failed_scenarios": 2,
        "pass_rate_pct": 0.0,
        "results": [
            {
                "id": "CALC_TEST",
                "title": "Test Calc Failure",
                "category": "calculation",
                "elapsed_sec": 1.23,
                "passed": False,
                "failure_reasons": ["Number mismatch"],
                "prompt": "Calculate 1 + 1",
                "number_audit": [
                    {
                        "expected": 2.0,
                        "in_tool": True,
                        "in_final": False,
                        "status": "MISSING",
                        "tool_diff": 0.0,
                        "final_diff": None,
                    }
                ],
                "tools_called": ["calculate_decimal"],
                "calc_tool_outputs": ["2.00"],
                "final_text": "Answer is 3",
            },
            {
                "id": "RAG_TEST",
                "title": "Test RAG Failure",
                "category": "rag_grounding",
                "elapsed_sec": 0.95,
                "passed": False,
                "failure_reasons": ["Missing statutory citation: CSSF 24/856"],
                "prompt": "Check regulatory limit",
                "number_audit": None,  # Must NOT crash when iterating
                "tools_called": ["search_documents"],
                "calc_tool_outputs": [],
                "expected_citations": ["CSSF 24/856"],
                "citations_ok": False,
                "expected_source_documents": ["CSSF24_856eng.pdf"],
                "matched_documents": [],
                "missing_documents": ["CSSF24_856eng.pdf"],
                "retrieval_ok": False,
                "final_text": "No citations provided",
            },
        ],
    }

    report_path = save_markdown_audit_report(fake_summary, output_dir=tmp_path)
    assert report_path.is_file()

    content = report_path.read_text(encoding="utf-8")
    assert "CALC_TEST" in content
    assert "Number Verification Audit Table:" in content
    assert "RAG_TEST" in content
    assert "RAG Retrieval & Citations Audit:" in content