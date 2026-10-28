"""Static assets (CSS/JS/HTML templates) and the pinned light theme for the chatbot UI."""

from pathlib import Path

import gradio as gr

from domain_profile import PROFILE

UI_DIR = Path(__file__).parent

# Load CSS template
CSS_PATH = UI_DIR / "static" / "css" / "style.css"
with open(CSS_PATH, "r", encoding="utf-8") as f:
    custom_css = f.read()

# Load JS template (input autofocus + reasoning-box state + forced light mode)
JS_PATH = UI_DIR / "static" / "js" / "focus.js"
with open(JS_PATH, "r", encoding="utf-8") as f:
    focus_js = f.read()

# Load Header HTML template
HEADER_PATH = UI_DIR / "templates" / "header.html"
with open(HEADER_PATH, "r", encoding="utf-8") as f:
    HEADER_HTML = f.read().format(
        profile_name=PROFILE.get("name", "RAG Assistant"),
        profile_description=PROFILE.get("description", "Document Assistant")
    )

# Load Chatbot Placeholder HTML template
PLACEHOLDER_PATH = UI_DIR / "templates" / "chatbot_placeholder.html"
with open(PLACEHOLDER_PATH, "r", encoding="utf-8") as f:
    CHATBOT_PLACEHOLDER = f.read().format(
        profile_name=PROFILE.get("name", "RAG Assistant"),
        profile_description=PROFILE.get("description", "Document Assistant")
    )

# Load Footer HTML template
FOOTER_PATH = UI_DIR / "templates" / "footer.html"
with open(FOOTER_PATH, "r", encoding="utf-8") as f:
    FOOTER_HTML = f.read()

# Load Profile pill template (sidebar bottom)
PROFILE_PATH = UI_DIR / "templates" / "profile.html"
with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    PROFILE_HTML = f.read()

# Floating expand button, shown only while the sidebar is collapsed
EXPAND_PATH = UI_DIR / "templates" / "expand.html"
with open(EXPAND_PATH, "r", encoding="utf-8") as f:
    EXPAND_HTML = f.read()

# Gemini-style plus menu template (source selection popup)
PLUS_MENU_PATH = UI_DIR / "templates" / "plus_menu.html"
with open(PLUS_MENU_PATH, "r", encoding="utf-8") as f:
    PLUS_MENU_HTML = f.read()

# --- THEME ---
# Light-only theme: every *_dark variant is pinned to its light value so the
# UI renders light even when the OS/browser prefers dark mode.
_LIGHT = dict(
    body_background_fill="#ffffff",
    body_text_color="#0f172a",
    body_text_color_subdued="#64748b",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f7f7f8",
    block_background_fill="#ffffff",
    block_border_color="#ececec",
    block_label_background_fill="#ffffff",
    block_label_text_color="#475569",
    block_title_text_color="#0f172a",
    input_background_fill="#f7f7f8",
    input_border_color="#ececec",
    button_primary_background_fill="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_text_color="#334155",
    button_cancel_background_fill="#ffffff",
    button_cancel_text_color="#334155",
    border_color_primary="#e2e8f0",
    border_color_accent="#1d4ed8",
    checkbox_background_color_selected="#eef2ff",
    checkbox_label_text_color="#334155",
    table_even_background_fill="#f8fafc",
    table_odd_background_fill="#ffffff",
    stat_background_fill="#ffffff",
    code_background_fill="#f6f8fa",
)
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=["-apple-system", "BlinkMacSystemFont", "San Francisco", "Helvetica Neue", "sans-serif"],
).set(
    block_shadow="0 1px 3px rgba(15, 23, 42, 0.08)",
    **_LIGHT,
    **{f"{k}_dark": v for k, v in _LIGHT.items()},
)
