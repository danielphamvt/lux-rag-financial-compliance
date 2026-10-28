"""FastAPI routes for the Evaluation & RAG Audit Dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ui.evaluation_history import (
    DATASET_PATH,
    DEFAULT_HISTORY_LIMIT,
    RESULTS_DIR,
    list_evaluation_history,
    load_evaluation_run,
)
# Re-export: existing importers (tests) pull this helper from ui.evaluation_routes.
from ui.evaluation_history import scenario_short_code as scenario_short_code
from ui.evaluation_run_manager import EVAL_MANAGER, EvaluationRunManager

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent
EVALUATION_TEMPLATE_PATH = UI_DIR / "templates" / "evaluation.html"

STREAM_POLL_INTERVAL_SEC = 0.15


def render_evaluation_page_html(
    base_html: str,
    manager: EvaluationRunManager,
    results_dir: Path,
    dataset_path: Path,
) -> str:
    """Hydrate evaluation.html with cache-busted asset URLs and the bootstrap payload."""
    css_path = UI_DIR / "static" / "css" / "evaluation.css"
    js_path = UI_DIR / "static" / "js" / "evaluation.js"
    v_css = int(css_path.stat().st_mtime) if css_path.is_file() else 1
    v_js = int(js_path.stat().st_mtime) if js_path.is_file() else 1

    rendered = base_html.replace(
        'href="/static/css/evaluation.css"',
        f'href="/static/css/evaluation.css?v={v_css}"',
    ).replace(
        'src="/static/js/evaluation.js"',
        f'src="/static/js/evaluation.js?v={v_js}"',
    )

    runs = list_evaluation_history(results_dir=results_dir, limit=DEFAULT_HISTORY_LIMIT)
    initial_run: dict[str, Any] | None = None

    if runs:
        active_filename = runs[0]["filename"]
        try:
            full_run = load_evaluation_run(
                active_filename, results_dir=results_dir, dataset_path=dataset_path
            )
            initial_run = {k: v for k, v in full_run.items() if k != "results"}
        except Exception as exc:
            logger.warning("Failed to pre-load initial run %s: %s", active_filename, exc)

    bootstrap_payload = {
        "runs": runs,
        "status": manager.get_status(),
        "initial_run": initial_run,
    }
    safe_json = (
        json.dumps(bootstrap_payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    bootstrap_script = f'<script id="eval-initial-state" type="application/json">{safe_json}</script>\n  '
    rendered = rendered.replace('<script src="/static/js/evaluation.js', bootstrap_script + '<script src="/static/js/evaluation.js')
    return rendered


def register_evaluation_routes(
    fastapi_app: FastAPI,
    evaluation_html: str | None = None,
    manager: EvaluationRunManager = EVAL_MANAGER,
    results_dir: Path = RESULTS_DIR,
    dataset_path: Path = DATASET_PATH,
) -> None:
    """Register HTML and REST/SSE endpoints for the Evaluation Dashboard on FastAPI.

    Args:
        fastapi_app: Target FastAPI application instance.
        evaluation_html: Optional HTML template string for `/evaluation`.
        manager: Subprocess execution manager instance.
        results_dir: Directory containing benchmark result files.
        dataset_path: Path to canonical 40-scenario dataset.
    """
    no_cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }

    @fastapi_app.get("/evaluation", response_class=HTMLResponse)
    def evaluation_page() -> HTMLResponse:
        raw_html = (
            evaluation_html
            if evaluation_html is not None
            else EVALUATION_TEMPLATE_PATH.read_text(encoding="utf-8")
        )
        content = render_evaluation_page_html(
            raw_html, manager=manager, results_dir=results_dir, dataset_path=dataset_path
        )
        return HTMLResponse(content, headers=no_cache_headers)

    @fastapi_app.get("/api/evaluation/history")
    def api_evaluation_history(limit: int = Query(default=DEFAULT_HISTORY_LIMIT, ge=1, le=20)) -> JSONResponse:
        runs = list_evaluation_history(results_dir=results_dir, limit=limit)
        return JSONResponse(
            content={
                "runs": runs,
                "status": manager.get_status(),
            },
            headers=no_cache_headers,
        )

    @fastapi_app.get("/api/evaluation/status")
    def api_evaluation_status() -> JSONResponse:
        return JSONResponse(content=manager.get_status(), headers=no_cache_headers)

    @fastapi_app.get("/api/evaluation/run/{filename}")
    def api_evaluation_run(filename: str) -> JSONResponse:
        data = load_evaluation_run(filename, results_dir=results_dir, dataset_path=dataset_path)
        return JSONResponse(content=data, headers=no_cache_headers)

    @fastapi_app.post("/api/evaluation/re-run")
    def api_evaluation_rerun() -> JSONResponse:
        started, status = manager.start_rerun()
        if not started:
            return JSONResponse(status_code=409, content={"started": False, **status})
        return JSONResponse(status_code=202, content={"started": True, **status})

    @fastapi_app.post("/api/evaluation/cancel")
    def api_evaluation_cancel() -> dict[str, Any]:
        return manager.cancel()

    @fastapi_app.get("/api/evaluation/stream")
    async def api_evaluation_stream(autostart: bool = Query(default=False)) -> StreamingResponse:
        if autostart and not manager.is_running():
            manager.start_rerun()

        async def _event_generator() -> AsyncIterator[str]:
            cursor = 0
            initial_status = manager.get_status()
            yield f"event: status\ndata: {json.dumps(initial_status)}\n\n"

            while True:
                lines, cursor = manager.get_logs_since(cursor)
                for line in lines:
                    payload = json.dumps({"line": line})
                    yield f"event: log\ndata: {payload}\n\n"

                status = manager.get_status()
                if not status["is_running"]:
                    final_lines, cursor = manager.get_logs_since(cursor)
                    for line in final_lines:
                        payload = json.dumps({"line": line})
                        yield f"event: log\ndata: {payload}\n\n"

                    recent = list_evaluation_history(results_dir=results_dir, limit=DEFAULT_HISTORY_LIMIT)
                    done_payload = {
                        **manager.get_status(),
                        "latest_filename": recent[0]["filename"] if recent else None,
                    }
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                    break

                await asyncio.sleep(STREAM_POLL_INTERVAL_SEC)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
