"""Scenario catalog and benchmark-run history helpers for the Evaluation Dashboard."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = WORKSPACE_ROOT / "evaluation" / "results" / "agent_reasoning"
DATASET_PATH = WORKSPACE_ROOT / "evaluation" / "eval_dataset_agent_reasoning.json"
FILENAME_PATTERN = re.compile(r"^agent_reasoning_benchmark(?:_[A-Za-z0-9_-]+)?\.json$")

DEFAULT_HISTORY_LIMIT = 5


def scenario_short_code(scenario_id: str, fallback_idx: int = 0) -> str:
    """Convert canonical scenario ID into compact matrix code (C1..C30, G1..G10).

    Args:
        scenario_id: Canonical scenario identifier (e.g. 'CALC_01', 'RAG_10').
        fallback_idx: Zero-based index used if ID does not match CALC/RAG prefix.

    Returns:
        Short badge code such as 'C1' or 'G10'.
    """
    sid = (scenario_id or "").strip().upper()
    calc_m = re.match(r"^CALC_0*(\d+)$", sid)
    if calc_m:
        return f"C{int(calc_m.group(1))}"
    rag_m = re.match(r"^RAG_0*(\d+)$", sid)
    if rag_m:
        return f"G{int(rag_m.group(1))}"
    return f"S{fallback_idx + 1}"


def format_run_timestamp_display(run_timestamp: str, iso_timestamp: str = "") -> str:
    """Format run_timestamp ('YYYY-MM-DD-HH-MM') into human-readable 'YYYY-MM-DD HH:MM'.

    Args:
        run_timestamp: Timestamp slug from benchmark JSON header.
        iso_timestamp: Fallback ISO-8601 timestamp string.

    Returns:
        Formatted timestamp string for UI display.
    """
    raw = (run_timestamp or "").strip()
    for fmt, out_fmt in (("%Y-%m-%d-%H-%M-%S", "%Y-%m-%d %H:%M:%S"), ("%Y-%m-%d-%H-%M", "%Y-%m-%d %H:%M")):
        try:
            return datetime.strptime(raw, fmt).strftime(out_fmt)
        except ValueError:
            pass
    if iso_timestamp:
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    return raw or "Unknown"


def load_scenario_catalog(dataset_path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load the canonical 40-scenario benchmark catalog from disk.

    Args:
        dataset_path: Path to eval_dataset_agent_reasoning.json.

    Returns:
        List of scenario definition dicts enriched with 'short_code'.
    """
    if not dataset_path.is_file():
        return []
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        scenarios = raw.get("scenarios", []) if isinstance(raw, dict) else raw
        catalog: list[dict[str, Any]] = []
        for idx, sc in enumerate(scenarios):
            sid = str(sc.get("id", f"SC_{idx + 1}"))
            catalog.append({
                "id": sid,
                "short_code": scenario_short_code(sid, idx),
                "title": sc.get("title", sid),
                "category": sc.get("category", "calculation"),
                "prompt": sc.get("prompt", ""),
                "ground_truth_numbers": sc.get("ground_truth_numbers", []),
                "tolerance": sc.get("tolerance", 0.01),
                "expected_citations": sc.get("expected_citations", []),
                "expected_source_documents": sc.get("expected_source_documents", []),
            })
        return catalog
    except Exception as exc:
        logger.error("Failed to read scenario catalog %s: %s", dataset_path, exc)
        return []


def list_evaluation_history(
    results_dir: Path = RESULTS_DIR,
    limit: int = DEFAULT_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """Scan evaluation results directory and return up to `limit` most recent runs.

    Args:
        results_dir: Directory containing agent_reasoning_benchmark_*.json files.
        limit: Maximum number of recent runs to return (default: 5).

    Returns:
        List of run summary dictionaries sorted newest-first.
    """
    if not results_dir.is_dir():
        return []

    candidates = sorted(results_dir.glob("agent_reasoning_benchmark_*.json"))
    if not candidates and (results_dir / "agent_reasoning_benchmark.json").is_file():
        candidates = [results_dir / "agent_reasoning_benchmark.json"]
    runs: list[dict[str, Any]] = []

    for file_path in candidates:
        if not file_path.is_file():
            continue
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipping unreadable benchmark file %s: %s", file_path.name, exc)
            continue

        results_list = data.get("results", [])
        total = int(data.get("total_scenarios", len(results_list)))
        passed = int(data.get("passed_scenarios", sum(1 for r in results_list if r.get("passed"))))
        failed = int(data.get("failed_scenarios", max(0, total - passed)))
        run_ts = str(data.get("run_timestamp") or file_path.stem.replace("agent_reasoning_benchmark_", ""))
        iso_ts = str(data.get("timestamp") or "")

        runs.append({
            "filename": file_path.name,
            "timestamp": iso_ts,
            "run_timestamp": run_ts,
            "display_timestamp": format_run_timestamp_display(run_ts, iso_ts),
            "total_scenarios": total,
            "passed_scenarios": passed,
            "failed_scenarios": failed,
            "pass_rate_pct": float(data.get("pass_rate_pct", 0.0)),
            "calculation_delegation_rate_pct": float(data.get("calculation_delegation_rate_pct", 0.0)),
            "numerical_accuracy_rate_pct": float(data.get("numerical_accuracy_rate_pct", 0.0)),
            "rag_grounding_rate_pct": float(data.get("rag_grounding_rate_pct", 0.0)),
            "rag_retrieval_fidelity_rate_pct": float(data.get("rag_retrieval_fidelity_rate_pct", 0.0)),
            "protocol_compliance_rate_pct": float(data.get("protocol_compliance_rate_pct", 0.0)),
            "avg_latency_sec": float(data.get("avg_latency_sec", 0.0)),
            "concurrency": int(data.get("concurrency", 1)),
            "_mtime": file_path.stat().st_mtime,
        })

    runs.sort(
        key=lambda r: (r["run_timestamp"], r["timestamp"], r["_mtime"], r["filename"]),
        reverse=True,
    )
    trimmed = runs[: max(1, limit)]
    for item in trimmed:
        item.pop("_mtime", None)
    return trimmed


def load_evaluation_run(
    filename: str,
    results_dir: Path = RESULTS_DIR,
    dataset_path: Path = DATASET_PATH,
) -> dict[str, Any]:
    """Load detailed JSON data for a specific benchmark run filename.

    Args:
        filename: Target JSON filename inside results_dir.
        results_dir: Directory containing benchmark result JSON files.
        dataset_path: Path to canonical 40-scenario dataset for matrix alignment.

    Returns:
        Full benchmark dictionary enriched with 'short_code' and 40-slot 'matrix'.

    Raises:
        HTTPException: If filename is invalid (400) or file does not exist (404).
    """
    clean_name = (filename or "").strip()
    if not FILENAME_PATTERN.match(clean_name):
        raise HTTPException(status_code=400, detail="Invalid evaluation filename.")

    resolved_dir = results_dir.resolve()
    file_path = (resolved_dir / clean_name).resolve()
    if not file_path.is_relative_to(resolved_dir) or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Evaluation run '{clean_name}' not found.")

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to parse evaluation JSON %s: %s", file_path, exc)
        raise HTTPException(status_code=500, detail=f"Corrupted evaluation JSON: {exc}") from exc

    raw_results: list[dict[str, Any]] = data.get("results", [])
    results_by_id: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(raw_results):
        sid = str(item.get("id", f"SC_{idx + 1}"))
        item["short_code"] = scenario_short_code(sid, idx)
        item["executed"] = True
        results_by_id[sid] = item

    catalog = load_scenario_catalog(dataset_path)
    matrix: list[dict[str, Any]] = []
    if catalog:
        for idx, base in enumerate(catalog):
            sid = base["id"]
            if sid in results_by_id:
                matrix.append(results_by_id[sid])
            else:
                matrix.append({
                    **base,
                    "short_code": scenario_short_code(sid, idx),
                    "executed": False,
                    "passed": None,
                    "elapsed_sec": 0.0,
                    "final_text": "",
                    "tool_delegation_ok": False,
                    "tools_called": [],
                    "tool_calls": [],
                    "calc_tool_outputs": [],
                    "search_tool_outputs": [],
                    "retrieved_documents": [],
                    "numbers_ok": False,
                    "number_audit": [],
                    "citations_ok": False,
                    "retrieval_ok": False,
                    "matched_documents": [],
                    "missing_documents": [],
                    "protocol": {},
                    "failure_reasons": ["Scenario was not executed in this partial benchmark run."],
                })
    else:
        matrix = list(raw_results)

    run_ts = str(data.get("run_timestamp") or file_path.stem.replace("agent_reasoning_benchmark_", ""))
    iso_ts = str(data.get("timestamp") or "")
    data["filename"] = file_path.name
    data["display_timestamp"] = format_run_timestamp_display(run_ts, iso_ts)
    data["matrix"] = matrix
    return data
