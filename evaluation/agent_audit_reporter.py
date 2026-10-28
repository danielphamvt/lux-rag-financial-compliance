"""
Agent Reasoning Audit Reporter & Failure Diagnostics.

Generates markdown audit reports and provides interactive terminal diagnostics
for analyzing why specific scenarios failed in the Tier 2/3 Benchmark.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "results" / "agent_reasoning"
DEFAULT_REPORTS_DIR = DEFAULT_RESULTS_DIR / "reports"
DEFAULT_BENCHMARK_PATH = DEFAULT_RESULTS_DIR / "agent_reasoning_benchmark.json"


def shorten_document_name(name: str) -> str:
    """Return concise canonical abbreviation for compliance documents."""
    n = name.lower()
    if "24_856" in n:
        return "CSSF24"
    if "02_77" in n:
        return "CSSF02"
    if "ucits" in n and "prospectus" not in n:
        return "UCITS"
    if "lux_alpha" in n or "prospectus" in n:
        return "LUX_ALPHA"
    return name.split(".")[0][:9]


def save_markdown_audit_report(
    summary: dict[str, Any], output_dir: Path = DEFAULT_REPORTS_DIR
) -> Path:
    """Save comprehensive markdown audit report with full traces for failed cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    report_file = output_dir / f"agent_reasoning_audit_{ts}.md"

    results = summary.get("results", [])
    total = summary.get("total_scenarios", len(results))
    passed = summary.get("passed_scenarios", sum(1 for r in results if r["passed"]))
    failed = [r for r in results if not r["passed"]]

    lines = [
        "# Tier 2/3 Agent Reasoning & Symbolic Calculation Audit Report",
        "",
        "## 1. Executive Summary",
        f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Europe/Paris)",
        f"- **Total Scenarios:** {total}",
        f"- **Passed Scenarios:** {passed} ({summary.get('pass_rate_pct', 0.0)}%)",
        f"- **Failed Scenarios:** {len(failed)}",
        f"- **Symbolic Delegation Rate (SDR):** {summary.get('calculation_delegation_rate_pct', 0.0)}%",
        f"- **Numerical Accuracy Rate (NAR):** {summary.get('numerical_accuracy_rate_pct', 0.0)}%",
        f"- **RAG Retrieval Fidelity Rate:** {summary.get('rag_retrieval_fidelity_rate_pct', 0.0)}%",
        f"- **Average Latency:** {summary.get('avg_latency_sec', 0.0)}s",
        "",
        "---",
        "",
        "## 2. Benchmark Scorecard",
        "",
        "| ID | Category | Latency | Tools | Num Match | RAG Docs Hit | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        v_str = "**PASS**" if r.get("passed") else "**FAIL**"
        if r.get("category") == "calculation" and r.get("number_audit"):
            matched_cnt = sum(1 for d in r["number_audit"] if d.get("matched"))
            tot_cnt = len(r["number_audit"])
            num_m = f"{matched_cnt}/{tot_cnt}"
        else:
            num_m = "-"

        if r.get("matched_documents"):
            m_list = [shorten_document_name(d) for d in (r.get("matched_documents") or [])]
            rag_m = ", ".join(m_list)
        else:
            rag_m = "-" if not r.get("missing_documents") else f"!{', '.join(shorten_document_name(d) for d in (r.get('missing_documents') or []))}"

        tools_str = ",".join(r.get("tools_called") or []) or "none"
        lines.append(
            f"| {r.get('id', '')} | {r.get('category', '')} | {r.get('elapsed_sec', 0.0)}s | {tools_str} | {num_m} | {rag_m} | {v_str} |"
        )

    lines.extend(["", "---", "", "## 3. Deep-Dive Audit for Failed Scenarios", ""])

    if not failed:
        lines.append("All scenarios passed! Zero failures detected.")
    else:
        for f in failed:
            lines.extend([
                f"### Scenario {f['id']}: {f['title']}",
                f"- **Category:** `{f['category']}`",
                f"- **Elapsed:** {f['elapsed_sec']}s",
                f"- **Verdict:** **FAIL**",
                "",
                "#### Failure Reasons:",
            ])
            for reas in f.get("failure_reasons") or []:
                lines.append(f"- ❌ {reas}")

            lines.extend([
                "",
                "#### Prompt:",
                f"> {f.get('prompt', 'N/A')}",
            ])

            if f.get("number_audit"):
                lines.extend([
                    "",
                    "#### Number Verification Audit Table:",
                    "| Expected | In Tool? | In Final? | Status | Tool Diff | Final Diff |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- |",
                ])
                for d in f["number_audit"]:
                    lines.append(
                        f"| {d['expected']} | {d.get('in_tool')} | {d.get('in_final')} | {d.get('status')} | {d.get('tool_diff')} | {d.get('final_diff')} |"
                    )

            if f.get("category") == "rag_grounding" or f.get("expected_citations") or f.get("expected_source_documents"):
                lines.extend([
                    "",
                    "#### RAG Retrieval & Citations Audit:",
                    f"- **Expected Citations:** `{f.get('expected_citations') or []}`",
                    f"- **Citations OK:** `{f.get('citations_ok', False)}`",
                    f"- **Expected Source Documents:** `{f.get('expected_source_documents') or []}`",
                    f"- **Matched Documents:** `{f.get('matched_documents') or []}`",
                    f"- **Missing Documents:** `{f.get('missing_documents') or []}`",
                    f"- **Retrieval OK:** `{f.get('retrieval_ok', False)}`",
                ])

            lines.extend([
                "",
                "#### Tools Invoked & Outputs:",
                f"- **Tools Called:** `{f.get('tools_called') or []}`",
            ])
            if f.get("calc_tool_outputs"):
                lines.append("```text")
                for out in f["calc_tool_outputs"]:
                    lines.append(str(out))
                lines.append("```")

            lines.extend([
                "",
                "#### Final Agent Response Text:",
                "```text",
                (f.get("final_text") or "").strip(),
                "```",
                "",
                "---",
                "",
            ])

    report_file.write_text("\n".join(lines), encoding="utf-8")
    return report_file


def explain_scenario_failure(
    scenario_id: str, benchmark_path: Path = DEFAULT_BENCHMARK_PATH
) -> None:
    """Print an interactive diagnostic breakdown of a failed scenario to the console."""
    if not benchmark_path.is_file():
        alt_path = WORKSPACE_ROOT / "evaluation" / "results" / "agent_reasoning_benchmark.json"
        if alt_path.is_file():
            benchmark_path = alt_path
        else:
            print(f"Error: Benchmark file not found at {benchmark_path}")
            return

    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    results = data.get("results", [])

    if scenario_id.lower() in ("all", "fails", "failed"):
        target_results = [r for r in results if not r.get("passed")]
    else:
        target_results = [r for r in results if r["id"].upper() == scenario_id.upper()]

    if not target_results:
        print(f"No scenario matching '{scenario_id}' found in {benchmark_path}")
        return

    for sc in target_results:
        print(f"\n{'='*95}")
        print(f"  AUDIT DIAGNOSTIC TRACE: {sc['id']} - {sc['title']}")
        print(f"{'='*95}")
        print(f"Category: {sc['category']} | Latency: {sc['elapsed_sec']}s | Verdict: {'PASS' if sc['passed'] else 'FAIL'}")
        print(f"\n--- PROMPT ---")
        print(sc.get("prompt", "N/A"))

        print(f"\n--- FAILURE REASONS ---")
        if sc.get("failure_reasons"):
            for reas in sc["failure_reasons"]:
                print(f"  ❌ {reas}")
        else:
            print("  None (Scenario Passed)")

        if sc.get("number_audit"):
            print(f"\n--- NUMERIC AUDIT DETAILS ---")
            print(f"{'Expected':<14} | {'In Tool':<9} | {'In Final':<9} | {'Status':<24} | {'ToolDiff':<10} | {'FinalDiff':<10}")
            print("-" * 85)
            for d in sc["number_audit"]:
                print(
                    f"{str(d['expected']):<14} | {str(d.get('in_tool')):<9} | {str(d.get('in_final')):<9} | "
                    f"{str(d.get('status')):<24} | {str(d.get('tool_diff')):<10} | {str(d.get('final_diff')):<10}"
                )

        if sc.get("category") == "rag_grounding" or sc.get("expected_citations") or sc.get("expected_source_documents"):
            print(f"\n--- RAG RETRIEVAL & CITATIONS AUDIT ---")
            print(f"Expected Citations: {sc.get('expected_citations') or []}")
            print(f"Citations OK: {sc.get('citations_ok', False)}")
            print(f"Expected Docs: {sc.get('expected_source_documents') or []}")
            print(f"Matched Docs: {sc.get('matched_documents') or []}")
            print(f"Missing Docs: {sc.get('missing_documents') or []}")
            print(f"Retrieval OK: {sc.get('retrieval_ok', False)}")

        print(f"\n--- TOOLS CALLED ---")
        print(sc.get("tools_called") or [])
        if sc.get("calc_tool_outputs"):
            print("--- CALCULATION TOOL OUTPUTS ---")
            for out in sc["calc_tool_outputs"]:
                print(out)

        print(f"\n--- FINAL AGENT ANSWER ---")
        print((sc.get("final_text") or "").strip())
        print(f"\n{'='*95}\n")
