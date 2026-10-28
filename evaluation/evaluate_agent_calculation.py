#!/usr/bin/env python3
"""
Tier 2 Evaluation Runner: End-to-End Agent Reasoning & Symbolic Calculation Benchmark.

Evaluates the LangGraph compliance agent across 40 scenarios:
- 30 Quantitative Symbolic Calculation & Delegation Scenarios (CALC_01–CALC_30)
- 10 Statutory RAG Grounding & Closed-World Abstention Scenarios (RAG_01–RAG_10)

Usage:
    python -m evaluation.evaluate_agent_calculation --smoke
    python -m evaluation.evaluate_agent_calculation --calc-only
    python -m evaluation.evaluate_agent_calculation --rag-only
    python -m evaluation.evaluate_agent_calculation --all
    python -m evaluation.evaluate_agent_calculation --mock
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

# Strictly enforce isolated test environment (.env.test) before loading any modules
from evaluation.eval_common import ensure_runtime

ensure_runtime()

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings
warnings.filterwarnings("ignore")

import logging
logging.captureWarnings(False)
logging.getLogger().setLevel(logging.ERROR)
for _log_name in (
    "httpx",
    "httpcore",
    "urllib3",
    "langchain",
    "langchain_core",
    "langgraph",
    "google",
    "chromadb",
    "asyncio",
):
    logging.getLogger(_log_name).setLevel(logging.ERROR)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = WORKSPACE_ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from evaluation.agent_audit_reporter import (  # noqa: E402
    explain_scenario_failure,
    save_markdown_audit_report,
    shorten_document_name,
)
from evaluation.agent_reasoning_verifier import evaluate_response  # noqa: E402

DATASET_PATH = WORKSPACE_ROOT / "evaluation" / "eval_dataset_agent_reasoning.json"
DEFAULT_RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "results" / "agent_reasoning"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "agent_reasoning_benchmark.json"


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load the 40-scenario benchmark dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("scenarios", []) if isinstance(raw, dict) else raw


def run_agent_scenario(
    scenario: dict[str, Any], use_mock: bool = False
) -> tuple[str, list[dict[str, Any]], list[str], list[str], float]:
    """Execute scenario on agent_graph or return deterministic mock."""
    if use_mock:
        return _mock_agent_scenario(scenario)

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from rag_agent.graph import agent_graph
    from rag_agent.nodes_edges import content_to_text

    prompt = scenario["prompt"]
    thread_id = f"eval_{scenario['id']}_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    t0 = time.perf_counter()
    state = agent_graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    elapsed = time.perf_counter() - t0

    text = content_to_text(state["messages"][-1].content) if state.get("messages") else ""
    tool_calls: list[dict[str, Any]] = []
    calc_tool_outputs: list[str] = []
    search_tool_outputs: list[str] = []

    # Primary evidence: the durable tool-execution ledger (appended at the real
    # execution points). The message-walk fallback serves legacy/rehydrated threads
    # whose ledger is empty; synthetic digest ids can never satisfy either path.
    ledger = state.get("tool_execution_log") or []
    for rec in ledger:
        tool_name = getattr(rec, "tool_name", "") or ""
        if tool_name in ("calculate_decimal", "run_nav_audit"):
            calc_tool_outputs.append(str(getattr(rec, "output", "")))
        elif tool_name in ("search_documents", "internet_search"):
            search_tool_outputs.append(str(getattr(rec, "output", "")))

    if ledger:
        # Tool-call provenance for delegation evidence comes from ledger args too,
        # so post-compression turns are indistinguishable from live ones.
        for rec in ledger:
            tool_calls.append({"name": getattr(rec, "tool_name", ""), "args": dict(getattr(rec, "args", None) or {})})
    else:
        # Gather from both agent_messages and messages to guarantee complete message history
        all_msgs = list(state.get("agent_messages", [])) + list(state.get("messages", []))
        seen_ids = set()

        for m in all_msgs:
            m_id = getattr(m, "id", None) or id(m)
            if m_id in seen_ids:
                continue
            seen_ids.add(m_id)

            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    if str(tc.get("id", "")).startswith("digest_"):
                        continue  # synthetic compressed-history stub, not real evidence
                    tool_calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
            elif isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
                if str(getattr(m, "id", "")).startswith("digest_"):
                    continue
                t_name = getattr(m, "name", "") or ""
                content_str = str(getattr(m, "content", ""))
                # Strict filtering: capture calculation outputs vs RAG retrieval outputs
                if t_name in ("calculate_decimal", "run_nav_audit"):
                    calc_tool_outputs.append(content_str)
                elif t_name in ("search_documents", "internet_search") or any(k in content_str for k in ("Document ID:", "File Name:")):
                    search_tool_outputs.append(content_str)
                elif not t_name:
                    calc_tool_outputs.append(content_str)

    return text, tool_calls, calc_tool_outputs, search_tool_outputs, elapsed


def _mock_agent_scenario(scenario: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[str], list[str], float]:
    """Produce deterministic mock response for schema and pipeline test."""
    sid = scenario["id"]
    cat = scenario.get("category", "calculation")
    raw_gt_nums = scenario.get("ground_truth_numbers", [])
    gt_nums = [
        (n["any_of"][0] if isinstance(n, dict) and "any_of" in n else (n[0] if isinstance(n, list) else n))
        for n in raw_gt_nums
    ]
    num_str = " ".join(str(n) for n in gt_nums)
    cits = scenario.get("expected_citations", ["CSSF 24/856"])
    exp_docs = scenario.get("expected_source_documents", ["CSSF24_856eng.pdf"])

    if cat == "calculation":
        tools = [{"name": "calculate_decimal", "args": {"expressions": [f"Result = {num_str}"]}}]
        calc_outputs = [f"- Result: `expr` => **{n}**" for n in gt_nums]
        search_outputs = [f"Parent ID: mock\nDocument ID: {d}\nFile Name: {d}\nSection / Page: Sec 1\nContent: mock" for d in exp_docs]
    elif scenario.get("is_abstention", False) or len(scenario.get("required_tools", [])) == 0:
        tools = []
        calc_outputs = []
        search_outputs = []
    else:
        tools = [{"name": "search_documents", "args": {"query": scenario["prompt"][:50]}}]
        calc_outputs = []
        search_outputs = [f"Parent ID: mock\nDocument ID: {d}\nFile Name: {d}\nSection / Page: Sec 1\nContent: mock" for d in exp_docs]

    text = (
        f"[REASONING]\nVerifying regulatory conditions and symbolic calculation.\n[CONCLUSION]\n"
        f"### CONCLUSION\nEvaluation for {sid} completed successfully. Figures: {num_str}.\n"
        f"### DETAILED ANALYSIS\nStatutory analysis grounds findings in {', '.join(cits)}.\n"
        f"---\n**SOURCES**\n- lux_alpha_ucits_prospectus.md\n- CSSF 24/856"
    )
    return text, tools, calc_outputs, search_outputs, 0.05


def run_benchmark(
    scenarios: list[dict[str, Any]],
    use_mock: bool = False,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    concurrency: int = 3,
) -> dict[str, Any]:
    """Run benchmark across selected scenarios and compile summary metrics."""
    results: list[dict[str, Any]] = []
    print_lock = threading.Lock()

    sep_header = "=" * 127
    sep_divider = "-" * 127

    print(f"\n{sep_header}\n  TIER 2/3: END-TO-END AGENT REASONING, SYMBOLIC CALCULATION & RAG BENCHMARK (Concurrency: {concurrency})\n{sep_header}")
    print(f"{'ID':<9} | {'Category':<14} | {'Elapsed':<9} | {'Tools Invoked':<35} | {'Num Match':<10} | {'RAG Docs Hit':<25} | {'Verdict':<7}")
    print(sep_divider)

    def _execute_single(sc: dict[str, Any]) -> dict[str, Any]:
        text, tool_calls, calc_tool_outputs, search_tool_outputs, elapsed = run_agent_scenario(sc, use_mock=use_mock)
        res = evaluate_response(sc, text, tool_calls, calc_tool_outputs, search_tool_outputs, elapsed)
        v_str = "PASS" if res["passed"] else "FAIL"

        # Format Tools Called: concise tool list
        tool_short = []
        for t in res.get("tools_called", []):
            if t == "calculate_decimal":
                tool_short.append("calc")
            elif t == "run_nav_audit":
                tool_short.append("audit")
            elif t == "search_documents":
                tool_short.append("search")
            else:
                tool_short.append(t[:6])
        tools_str = ",".join(tool_short) if tool_short else "none"

        # Format Num Match: concise matched / total ratio (only for calculation scenarios)
        if sc.get("category") == "calculation" and res.get("number_audit"):
            matched_cnt = sum(1 for d in res["number_audit"] if d["matched"])
            tot_cnt = len(res["number_audit"])
            num_m = f"{matched_cnt}/{tot_cnt}"
        else:
            num_m = "-"

        # Format RAG Docs Hit: concise document names
        exp_docs = sc.get("expected_source_documents", [])
        if not exp_docs:
            rag_m = "-"
        elif res.get("retrieval_ok"):
            m_list = [shorten_document_name(d) for d in res.get("matched_documents", [])]
            rag_m = ",".join(m_list) or "-"
        else:
            miss_list = [f"!{shorten_document_name(d)}" for d in res.get("missing_documents", [])]
            rag_m = ",".join(miss_list)

        elapsed_str = f"{res['elapsed_sec']:.2f} s"
        with print_lock:
            print(f"{sc['id']:<9} | {sc['category']:<14} | {elapsed_str:<9} | {tools_str:<35} | {num_m:<10} | {rag_m:<25} | {v_str:<7}")

        return res

    if concurrency > 1 and len(scenarios) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {executor.submit(_execute_single, sc): idx for idx, sc in enumerate(scenarios)}
            indexed_results: list[dict[str, Any] | None] = [None] * len(scenarios)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                indexed_results[idx] = future.result()
            results = [r for r in indexed_results if r is not None]
    else:
        for sc in scenarios:
            results.append(_execute_single(sc))

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    calc_results = [r for r in results if r["category"] == "calculation"]
    rag_results = [r for r in results if r["category"] == "rag_grounding"]
    scs_with_docs = [r for r in results if r.get("matched_documents") or r.get("missing_documents")]

    sdr = (sum(1 for r in calc_results if r["tool_delegation_ok"]) / len(calc_results) * 100.0) if calc_results else 100.0
    nar = (sum(1 for r in calc_results if r["numbers_ok"]) / len(calc_results) * 100.0) if calc_results else 100.0
    rcgr = (sum(1 for r in rag_results if r["citations_ok"]) / len(rag_results) * 100.0) if rag_results else 100.0
    rf_pct = (sum(1 for r in scs_with_docs if r["retrieval_ok"]) / len(scs_with_docs) * 100.0) if scs_with_docs else 100.0
    pcr = (sum(1 for r in results if r["protocol"]["protocol_passed"]) / total * 100.0) if total else 100.0
    avg_lat = sum(r["elapsed_sec"] for r in results) / total if total else 0.0

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H-%M")
    timestamp_iso = now.isoformat()

    summary = {
        "timestamp": timestamp_iso,
        "run_timestamp": timestamp_str,
        "total_scenarios": total,
        "passed_scenarios": passed_count,
        "failed_scenarios": total - passed_count,
        "pass_rate_pct": round(passed_count / total * 100.0, 2) if total else 0.0,
        "calculation_delegation_rate_pct": round(sdr, 2),
        "numerical_accuracy_rate_pct": round(nar, 2),
        "rag_grounding_rate_pct": round(rcgr, 2),
        "rag_retrieval_fidelity_rate_pct": round(rf_pct, 2),
        "protocol_compliance_rate_pct": round(pcr, 2),
        "avg_latency_sec": round(avg_lat, 2),
        "concurrency": concurrency,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(summary, ensure_ascii=False, indent=2)
    output_path.write_text(json_payload, encoding="utf-8")

    # Save timestamped JSON archive for future audit & web UI display
    timestamped_json = output_path.parent / f"agent_reasoning_benchmark_{timestamp_str}.json"
    timestamped_json.write_text(json_payload, encoding="utf-8")

    report_file = save_markdown_audit_report(summary)
    print(sep_divider)
    print(f"Overall Pass Rate: {summary['pass_rate_pct']}% ({passed_count}/{total}) | SDR: {sdr:.1f}% | NAR: {nar:.1f}% | Avg Latency: {avg_lat:.2f}s")
    print(f"Saved latest JSON summary to: {output_path}")
    print(f"Saved timestamped audit JSON to: {timestamped_json}")
    print(f"Detailed Markdown Audit Trail saved to: {report_file}")
    print(f"{sep_header}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tier 2 Agent Reasoning Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run 2-scenario smoke test (CALC_01 + RAG_01)")
    parser.add_argument("--calc-only", action="store_true", help="Run only 30 calculation scenarios")
    parser.add_argument("--rag-only", action="store_true", help="Run only 10 RAG scenarios")
    parser.add_argument("--all", action="store_true", help="Run all 40 scenarios")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock for fast testing")
    parser.add_argument("--concurrency", "-c", type=int, default=3, help="Concurrent worker threads (default: 3)")
    parser.add_argument("--scenario", type=str, default="", help="Run single scenario ID (e.g. CALC_01)")
    parser.add_argument("--explain", type=str, default="", help="Diagnose failure reasons for scenario ID (e.g. CALC_08 or 'fails')")
    parser.add_argument("--env-file", type=str, default=".env.test", help="Test environment configuration file")
    args = parser.parse_args()

    os.environ["ENV_FILE"] = args.env_file

    if args.explain:
        explain_scenario_failure(args.explain, DEFAULT_OUTPUT_PATH)
        return

    scenarios = load_dataset()

    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
    elif args.smoke:
        scenarios = [s for s in scenarios if s["id"] in ("CALC_01", "RAG_01")]
    elif args.calc_only:
        scenarios = [s for s in scenarios if s["category"] == "calculation"]
    elif args.rag_only:
        scenarios = [s for s in scenarios if s["category"] == "rag_grounding"]

    if not scenarios:
        print("No scenarios matched the given criteria.")
        sys.exit(1)

    run_benchmark(scenarios, use_mock=args.mock, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
