import logging

import gradio as gr

from db import chat_store
from domain_profile import PROFILE
from ui.chatbot_theme import (
    CHATBOT_PLACEHOLDER,
    EXPAND_HTML,
    FOOTER_HTML,
    HEADER_HTML,
    PLUS_MENU_HTML,
    PROFILE_HTML,
    custom_css,
    focus_js,
    theme,
)

logger = logging.getLogger(__name__)

# Featured Demo Queries loaded directly from the domain profile (Single Source of Truth)
EXAMPLES = PROFILE.get("sample_questions", [])


def create_demo(chat_fn):
    head_script = f"""<script>
{focus_js}
</script>"""
    with gr.Blocks(
        title="Second-Level NAV Compliance & Audit RAG",
        theme=theme,
        css=custom_css,
        head=head_script,
        fill_height=True,
        fill_width=True
    ) as demo:

        # Conversation state: active conversation id + a counter that bumps to
        # re-render the sidebar history list
        conv_id_state = gr.State(None)
        conv_refresh = gr.State(0)

        with gr.Row(elem_id="app-shell"):
            # --- Left sidebar: brand, new chat, skill selector, sample questions,
            #     searchable persisted history, footer, profile ---
            with gr.Column(elem_id="sidebar", scale=0, min_width=0):
                gr.HTML(HEADER_HTML)

                new_chat_btn = gr.Button("New Audit Case", elem_id="new-chat-btn")

                with gr.Accordion("Sample Questions", open=False, elem_id="sample-questions"):
                    with gr.Column(elem_classes="sidebar-list"):
                        example_buttons = [
                            gr.Button(ex, elem_classes="sidebar-item") for ex in EXAMPLES
                        ]

                gr.HTML('<div class="sidebar-label">Chat History</div>')
                conv_search = gr.Textbox(
                    placeholder="Search conversations...",
                    show_label=False,
                    container=False,
                    elem_id="conv-search",
                )

                # --- Persisted conversation history (newest first, capped by the store) ---
                # chatbot/history below are late-bound: this renders at app load,
                # after the whole Blocks layout exists.
                @gr.render(inputs=[conv_refresh, conv_search, conv_id_state])
                def recent_conversations(_tick, query, active_cid):
                    with gr.Column(elem_classes="sidebar-list conv-list"):
                        items = chat_store.list_recent()
                        if query and query.strip():
                            q = query.strip().lower()
                            items = [(t, c) for (t, c) in items if q in t.lower()]
                        if not items:
                            empty = "No matching conversations" if (query and query.strip()) else "No conversations yet"
                            gr.HTML(f'<div class="conv-empty">{empty}</div>')
                        for title, cid in items:
                            is_active = (active_cid == cid)
                            row_classes = "conv-row active" if is_active else "conv-row"
                            with gr.Row(elem_classes=row_classes):
                                title_btn = gr.Button(title, elem_classes="sidebar-item conv-title", scale=1, min_width=0)
                                title_btn.click(
                                    fn=lambda c=cid: _load_messages(c),
                                    outputs=[chatbot, history, conv_id_state],
                                )
                                del_btn = gr.Button(
                                    "",
                                    elem_classes="conv-delete",
                                    scale=0,
                                    min_width=0,
                                    size="sm",
                                )
                                del_btn.click(
                                    fn=lambda r, cur, c=cid: _delete_conversation(r, cur, c),
                                    inputs=[conv_refresh, conv_id_state],
                                    outputs=[conv_refresh, chatbot, history, conv_id_state],
                                )

                gr.HTML(FOOTER_HTML)
                gr.HTML(PROFILE_HTML)

            # --- Main column: chat + input ---
            with gr.Column(elem_id="main", scale=1):
                gr.HTML(EXPAND_HTML)

                chatbot = gr.Chatbot(
                    elem_id="chat-box",
                    show_label=False,
                    placeholder=CHATBOT_PLACEHOLDER,
                    avatar_images=(None, None),
                    scale=1,
                    type="messages",
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "\\[", "right": "\\]", "display": True},
                        {"left": "\\(", "right": "\\)", "display": False},
                    ],
                )

                with gr.Column(elem_id="custom-input-container", scale=0):
                    with gr.Row(elem_classes="input-row"):
                        plus_menu = gr.HTML(PLUS_MENU_HTML, elem_id="input-plus-menu-wrapper")
                        command_chip = gr.HTML('<div id="command-chip-container" style="display: none;"></div>', elem_id="command-chip-wrapper")
                        chat_input = gr.Textbox(
                            placeholder="Ask anything about your document corpus...",
                            scale=7,
                            autofocus=False,
                            show_label=False,
                            container=False,
                            elem_id="chat-input"
                        )
                        submit_btn = gr.Button("", elem_id="submit-btn", scale=0, min_width=38)
                        stop_btn = gr.Button("", elem_id="stop-btn", scale=0, min_width=38, visible=False)
                    source_mode = gr.Textbox(value="RAG only", visible=False, elem_id="selected-source-mode")
                    gr.HTML(
                        '<div class="input-helper-note">'
                        '💡 Audit Practice: Start a New Audit Case when switching circulars/questions to maintain isolated statutory context.'
                        '</div>'
                    )

        # 3. State and event wiring
        history = gr.State([])

        def _load_messages(cid):
            msgs = chat_store.get_messages(cid)
            return msgs, msgs, cid

        def _delete_conversation(refresh, current, cid):
            try:
                chat_store.delete_conversation(cid)
            except Exception as e:
                logger.warning("Failed to delete conversation %s: %s", cid, e)
            if current == cid:
                # Deleted the open conversation: fall back to a fresh chat
                return refresh + 1, [], [], None
            return refresh + 1, gr.update(), gr.update(), gr.update()

        new_chat_btn.click(
            fn=lambda: ([], [], None, "RAG only"),
            outputs=[chatbot, history, conv_id_state, source_mode],
            js="() => { if (window.resetSourceMode) window.resetSourceMode(); }"
        )

        # Sidebar sample questions insert into the input box
        for btn, ex in zip(example_buttons, EXAMPLES):
            btn.click(fn=lambda x=ex: x, outputs=[chat_input])


        # Show/hide the send button & expand input height (>2x) when >= 2 lines
        toggle_btn_js = """
        function(text) {
            var ta = document.querySelector('#chat-input textarea');
            var container = document.getElementById('custom-input-container');
            var btn = document.getElementById('submit-btn');
            var val = (typeof text === 'string') ? text : (ta ? ta.value : '');

            if (btn) {
                if (!val || val.trim() === '') {
                    btn.style.opacity = '0.5';
                    btn.style.pointerEvents = 'none';
                } else {
                    btn.style.opacity = '1';
                    btn.style.pointerEvents = 'auto';
                }
            }

            if (window.syncChatInputUI) {
                window.syncChatInputUI(val, false);
                return;
            }

            if (!ta || !container) return;
            var row = container.querySelector('.input-row');
            var lineBreaks = (val.match(/\\r?\\n/g) || []).length;
            var w = ta.clientWidth || ta.offsetWidth || 620;
            var charsPerLine = Math.max(40, Math.floor((w - 20) / 7.2));
            var isMulti = (lineBreaks >= 1) || (val.length > charsPerLine);

            if (isMulti) {
                container.classList.add('is-multiline');
                document.body.classList.add('chat-input-multiline');
                if (row) {
                    row.style.setProperty('min-height', '116px', 'important');
                    row.style.setProperty('height', 'auto', 'important');
                    row.style.setProperty('border-radius', '20px', 'important');
                }
                ta.style.setProperty('height', '104px', 'important');
                ta.style.setProperty('min-height', '104px', 'important');
                ta.style.setProperty('max-height', '180px', 'important');
                ta.style.setProperty('overflow-y', 'auto', 'important');
            } else {
                container.classList.remove('is-multiline');
                document.body.classList.remove('chat-input-multiline');
                if (row) {
                    row.style.setProperty('min-height', '48px', 'important');
                    row.style.setProperty('height', '48px', 'important');
                    row.style.setProperty('border-radius', '26px', 'important');
                }
                ta.style.setProperty('height', '40px', 'important');
                ta.style.setProperty('min-height', '40px', 'important');
                ta.style.setProperty('max-height', '40px', 'important');
                ta.style.setProperty('overflow-y', 'hidden', 'important');
            }
        }
        """
        chat_input.change(fn=None, inputs=[chat_input], outputs=[], js=toggle_btn_js)

        def user_act(user_msg, hist, conv_id):
            if not (user_msg or "").strip():
                return user_msg or "", hist, conv_id, gr.update(), gr.update()

            # Preserve raw user text cleanly without prepending mode slash commands
            text = user_msg

            if conv_id is None:
                try:
                    conv_id = chat_store.create_conversation(chat_store.make_title(text))
                except Exception as e:
                    logger.warning("Failed to create conversation: %s", e)
            if conv_id is not None:
                try:
                    chat_store.add_message(conv_id, "user", text, plain=text)
                except Exception as e:
                    logger.warning("Failed to persist user turn: %s", e)

            hist.append({"role": "user", "content": text})
            hist.append({"role": "assistant", "content": ""})
            # Swap Send for Stop while the reply streams
            return "", hist, conv_id, gr.update(visible=True), gr.update(visible=False)

        def bot_act(hist, conv_id, mode="RAG only"):
            # A pending turn always ends with an empty assistant message; if it
            # does not, the submit was a no-op (e.g. blank input)
            if not hist or hist[-1]["content"] != "":
                yield hist
                return

            user_msg = hist[-2]["content"]
            stream = chat_fn(user_msg, hist[:-2], conv_id, mode)
            # chat_fn yields response string chunk by chunk
            response = ""
            for chunk in stream:
                response = chunk
                hist[-1]["content"] = response
                yield hist

        def restore_buttons():
            return gr.update(visible=False), gr.update(visible=True)

        def stop_generation(hist, refresh):
            # Cancelled mid-stream: drop the pending empty assistant marker so
            # no blank reply row is left behind, then restore Send
            if hist and hist[-1].get("role") == "assistant" and hist[-1].get("content") == "":
                hist = hist[:-1]
            return hist, hist, gr.update(visible=False), gr.update(visible=True), refresh + 1

        # Wire Enter key and Send button; after each exchange, bump the counter
        # so the sidebar history list re-renders with the latest conversations
        act_outputs = [chat_input, history, conv_id_state, stop_btn, submit_btn]
        sub_evt1 = chat_input.submit(user_act, [chat_input, history, conv_id_state], act_outputs, queue=False).then(
            bot_act, [history, conv_id_state, source_mode], chatbot
        ).then(restore_buttons, None, [stop_btn, submit_btn])
        sub_evt2 = submit_btn.click(user_act, [chat_input, history, conv_id_state], act_outputs, queue=False).then(
            bot_act, [history, conv_id_state, source_mode], chatbot
        ).then(restore_buttons, None, [stop_btn, submit_btn])
        sub_evt1.then(lambda n: n + 1, conv_refresh, conv_refresh)
        sub_evt2.then(lambda n: n + 1, conv_refresh, conv_refresh)
        stop_btn.click(
            stop_generation,
            [history, conv_refresh],
            [chatbot, history, stop_btn, submit_btn, conv_refresh],
            cancels=[sub_evt1, sub_evt2],
        )

    return demo
