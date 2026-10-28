import logging
import warnings

import gradio as gr

# Silence noisy downstream library warnings (transformers/torch) without hiding real errors
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from pathlib import Path

from config import GRADIO_SERVER_NAME, GRADIO_SERVER_PORT



from ui.chatbot_ui import create_demo
from ui.renderers import UI_DIR
from ui.stream_handler import _chat_stream
from ui.evaluation_routes import register_evaluation_routes
from ui.renderers import ERROR_NOTICE, get_display_html

def chat(message, history, conv_id, source_mode: str = "RAG only"):
    """Gradio entry point: stream the response, turning system errors into friendly notices."""
    try:
        yield from _chat_stream(message, history, conv_id, source_mode=source_mode)
    except Exception:
        logger.exception("Chat pipeline failed")
        yield get_display_html("", ERROR_NOTICE, "final", is_completed=True)


# Load Architecture page template (served at /architecture for demos)
with open(UI_DIR / "templates" / "architecture.html", "r", encoding="utf-8") as f:
    ARCHITECTURE_BODY = f.read()


def create_app():
    """Build FastAPI app: /architecture page + static assets, chatbot mounted at /."""
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles

    fastapi_app = FastAPI()

    # Static assets (mermaid.min.js for offline diagram rendering)
    fastapi_app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

    @fastapi_app.get("/prospectus/{filename}")
    def download_compliance_document(filename: str):
        samples_dir = (UI_DIR.parent.parent / "data_docs").resolve()
        file_path = (samples_dir / filename).resolve()
        if not file_path.is_relative_to(samples_dir) or not file_path.is_file():
            raise HTTPException(status_code=404, detail="Document not found")
        media_type = "text/markdown; charset=utf-8" if filename.endswith(".md") else "application/pdf"
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
        )

    @fastapi_app.get("/manifest.json")
    def web_manifest():
        return {
            "name": "NAV Compliance & Audit RAG",
            "short_name": "NAV Audit RAG",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#1d4ed8",
        }

    @fastapi_app.get("/architecture", response_class=HTMLResponse)
    def architecture_page():
        # Compose a standalone page: diagrams render client-side from local mermaid.min.js
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic Document RAG · System Architecture</title>
<link rel="stylesheet" href="/static/css/architecture.css">
</head>
<body>
<a class="arch-back" href="/">&#8592; Back to Chat</a>
{ARCHITECTURE_BODY}
<script src="/static/js/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'neutral', securityLevel: 'loose', flowchart: {{ useMaxWidth: true }} }});</script>
</body>
</html>"""
        return HTMLResponse(html)

    register_evaluation_routes(fastapi_app)

    demo = create_demo(chat)
    # Chatbot at root; explicit /architecture and /static routes registered above take precedence
    fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    return fastapi_app, uvicorn


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    fastapi_app, uvicorn = create_app()
    try:
        from rag_agent.tools import warm_up_backend

        logger = logging.getLogger(__name__)
        warmed = warm_up_backend()
        if warmed:
            logger.info("Backend warm-up complete — models loaded, ready to serve.")
        else:
            logger.info("Backend 'page_index' ready to serve (vectorless — zero model warm-up overhead).")
    except Exception as exc:  # noqa: BLE001 — UI must still start if the index is unavailable
        print(f"Backend warm-up failed ({exc}); the first query will load models on demand.")

    uvicorn.run(fastapi_app, host=GRADIO_SERVER_NAME, port=GRADIO_SERVER_PORT)
