"""Standalone tests for the Evaluation & RAG Audit Dashboard (/evaluation)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

UI_DIR = Path(__file__).resolve().parent.parent / "frontend" / "ui"




def test_evaluation_api_history_and_run_loading(tmp_path):
    """Verify history sorting (top 5) and 40-slot matrix loading."""
    from ui.evaluation_routes import (
        EvaluationRunManager,
        register_evaluation_routes,
        scenario_short_code,
    )

    assert scenario_short_code("CALC_01") == "C1"
    assert scenario_short_code("CALC_30") == "C30"
    assert scenario_short_code("RAG_01") == "G1"
    assert scenario_short_code("RAG_10") == "G10"

    for i in range(1, 7):
        ts = f"2026-10-25-12-0{i}"
        payload = {
            "timestamp": f"2026-10-25T12:0{i}:00",
            "run_timestamp": ts,
            "total_scenarios": 40,
            "passed_scenarios": 34 + i,
            "failed_scenarios": 6 - i,
            "pass_rate_pct": round((34 + i) / 40 * 100.0, 2),
            "calculation_delegation_rate_pct": 100.0,
            "numerical_accuracy_rate_pct": 96.67,
            "rag_grounding_rate_pct": 100.0,
            "rag_retrieval_fidelity_rate_pct": 100.0,
            "protocol_compliance_rate_pct": 97.5,
            "avg_latency_sec": 20.0 + i,
            "concurrency": 3,
            "results": [
                {
                    "id": "CALC_01",
                    "title": "Class R Flat Fee Accrual",
                    "category": "calculation",
                    "elapsed_sec": 12.3,
                    "prompt": "Compute Class R fee",
                    "final_text": "[REASONING] Step 1\n[CONCLUSION]\n### CONCLUSION\n110,958.90 EUR",
                    "tools_called": ["search_documents", "calculate_decimal"],
                    "passed": True,
                    "failure_reasons": [],
                }
            ],
        }
        (tmp_path / f"agent_reasoning_benchmark_{ts}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    app = FastAPI()
    mgr = EvaluationRunManager(workspace_root=tmp_path)
    register_evaluation_routes(app, manager=mgr, results_dir=tmp_path)
    client = TestClient(app)

    page_resp = client.get("/evaluation")
    assert page_resp.status_code == 200
    assert "Agent Reasoning &amp; Compliance Evaluation" in page_resp.text
    assert "2026-10-25 12:06" in page_resp.text
    assert "2026-10-25 12:02" in page_resp.text
    assert "Class R Flat Fee Accrual" in page_resp.text
    assert 'id="eval-initial-state"' in page_resp.text
    assert "/static/js/evaluation.js?v=" in page_resp.text

    hist_resp = client.get("/api/evaluation/history")
    assert hist_resp.status_code == 200
    runs = hist_resp.json()["runs"]
    assert len(runs) == 5
    assert runs[0]["run_timestamp"] == "2026-10-25-12-06"
    assert runs[-1]["run_timestamp"] == "2026-10-25-12-02"
    assert runs[0]["display_timestamp"] == "2026-10-25 12:06"

    run_resp = client.get("/api/evaluation/run/agent_reasoning_benchmark_2026-10-25-12-06.json")
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["pass_rate_pct"] == 100.0
    assert len(run_data["matrix"]) == 40
    assert run_data["matrix"][0]["short_code"] == "C1"
    assert run_data["matrix"][0]["executed"] is True
    assert run_data["matrix"][39]["short_code"] == "G10"

    bad_resp = client.get("/api/evaluation/run/../../etc/passwd")
    assert bad_resp.status_code in (400, 404)
    missing_resp = client.get("/api/evaluation/run/agent_reasoning_benchmark_2099-01-01.json")
    assert missing_resp.status_code == 404


def test_evaluation_rerun_subprocess_and_sse_stream(tmp_path):
    """Verify duplicate run lock (409 Conflict) and real-time SSE stdout streaming."""
    from ui.evaluation_routes import EvaluationRunManager, register_evaluation_routes

    mgr = EvaluationRunManager(workspace_root=tmp_path, timeout_sec=10)
    app = FastAPI()
    register_evaluation_routes(app, evaluation_html="<html></html>", manager=mgr, results_dir=tmp_path)
    client = TestClient(app)

    started, status = mgr.start_rerun(
        command=[
            sys.executable,
            "-u",
            "-c",
            "import time; print('LINE_1', flush=True); time.sleep(0.35); print('LINE_2', flush=True)",
        ]
    )
    assert started is True
    assert status["is_running"] is True

    dup_resp = client.post("/api/evaluation/re-run")
    assert dup_resp.status_code == 409
    assert dup_resp.json()["started"] is False
    assert dup_resp.json()["is_running"] is True

    with client.stream("GET", "/api/evaluation/stream") as stream_resp:
        assert stream_resp.status_code == 200
        body = "".join(stream_resp.iter_text())
    assert "LINE_1" in body
    assert "LINE_2" in body
    assert "event: done" in body

    time.sleep(0.05)
    final_status = client.get("/api/evaluation/status").json()
    assert final_status["is_running"] is False
    assert final_status["status"] == "completed"
