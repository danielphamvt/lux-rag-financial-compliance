"""Tests for UI assets, warning disclaimers, and dynamic backend interfaces."""
from pathlib import Path
UI_DIR = Path(__file__).resolve().parent.parent / "frontend" / "ui"


def test_yellow_warning_triangle_svg_and_external_disclaimer_retention():
    from ui.renderers import clean_display_sources, get_display_html
    from rag_agent.harness import AI_DISCLAIMER

    # 1. 100% Corpus response: strips AI_DISCLAIMER and emits no yellow warning triangle SVG
    corpus_text = (
        f"### CONCLUSION\n\nCompliant under Circular CSSF 24/856.\n\n"
        f"### DETAILED ANALYSIS\n\nAnalysis text.\n\n{AI_DISCLAIMER}\n\n"
        f"---\n**SOURCES**\n- Circular CSSF 24/856"
    )
    corpus_html = clean_display_sources(get_display_html("", corpus_text, "final", is_completed=True))
    assert "warning-triangle-icon" not in corpus_html
    assert AI_DISCLAIMER not in corpus_html

    # 2. External source (Internet Search / AI's Built-in Knowledge / Hybrid): renders yellow warning triangle SVG & keeps disclaimer
    for warning_hdr in (
        "⚠️ **WARNING — Data Source: Internet Search (External Web Reference):** External web answer.",
        "⚠️ **WARNING — Data Source: AI's Built-in Knowledge:** Parametric knowledge answer.",
        "⚠️ **WARNING — Data Source: Hybrid (Internal Compliance Corpus + Internet Search):** Hybrid answer.",
    ):
        ext_text = (
            f"### CONCLUSION\n\n{warning_hdr}\n\n"
            f"### DETAILED ANALYSIS\n\nAnalysis text.\n\n{AI_DISCLAIMER}\n\n"
            f"---\n**SOURCES**\n- Not found in the provided document database; the answer is based on the AI's built-in knowledge."
        )
        ext_html = clean_display_sources(get_display_html("", ext_text, "final", is_completed=True))
        assert 'class="warning-triangle-icon"' in ext_html
        assert 'fill="#f59e0b"' in ext_html
        assert warning_hdr in ext_html
        assert AI_DISCLAIMER in ext_html


def test_rag_telemetry_artifact_and_reasoning_card_rendering(monkeypatch):
    from ui.renderers import format_rag_telemetry_html, format_reasoning_html, get_display_html
    from rag_engine.core.schemas import RetrievalResult, RetrievedChunk
    import rag_agent.tools as tools_mod

    class _DummyBackend:
        name = "dummy"

        def retrieve(self, query: str, audit_date: str | None = None, top_k: int = 5) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                audit_date=audit_date,
                chunks=[
                    RetrievedChunk(
                        id="p1",
                        text="Section 5.2 tolerance rules",
                        score=0.891,
                        doc_id="CSSF_24_856",
                        source_file="CSSF24_856eng.pdf",
                        section_id="Section 5.2",
                        page_number=28,
                        valid_from="2025-01-01",
                        valid_to="9999-12-31",
                    ),
                    RetrievedChunk(
                        id="p2",
                        text="Article 52 concentration limits",
                        score=0.814,
                        doc_id="UCITS_2009_65_EC",
                        source_file="UCITS_Directive_2009_65_EC.pdf",
                        section_id="Article 52",
                        page_number=None,
                        valid_from="2009-11-17",
                        valid_to="9999-12-31",
                    ),
                ],
            )

    monkeypatch.setattr(tools_mod, "_get_backend", lambda: _DummyBackend())

    # 1. Direct invoke returns plain string (zero telemetry bloat in LLM context)
    plain_out = tools_mod.search_documents.invoke(
        {"query": "UCITS NAV pricing error threshold calculation", "audit_date": "2025-02-15"}
    )
    assert isinstance(plain_out, str)
    assert "Section 5.2 tolerance rules" in plain_out
    assert "rag_telemetry" not in plain_out

    # 2. ToolCall invoke returns ToolMessage with structured out-of-band artifact
    tool_msg = tools_mod.search_documents.invoke({
        "name": "search_documents",
        "args": {"query": "UCITS NAV pricing error threshold calculation", "audit_date": "2025-02-15"},
        "id": "tc_rag_1",
        "type": "tool_call",
    })
    assert tool_msg.content == plain_out
    artifact = tool_msg.artifact
    assert isinstance(artifact, dict)
    assert artifact["type"] == "rag_telemetry"
    assert artifact["latency_ms"] >= 1
    assert artifact["chunk_count"] == 2
    assert "audit_date = 2025-02-15 (Circular CSSF 24/856)" in artifact["filter_regime"]

    # 3. Reasoning HTML renders the card intact alongside reasoning bullets
    card_html = format_rag_telemetry_html(artifact)
    combined = format_reasoning_html(f"{card_html}\n- Verify CSSF 24/856 threshold")
    assert "RAG Retrieval" in combined
    assert "2 chunks retrieved" in combined
    assert "UCITS NAV pricing error threshold calculation" in combined
    assert "audit_date = 2025-02-15 (Circular CSSF 24/856)" in combined
    assert "[Score: 0.89]" in combined
    assert "CSSF 24/856 — Section 5.2 (Page 28)" in combined
    assert "[Score: 0.81]" in combined
    assert "UCITS Directive 2009/65/EC — Article 52" in combined
    assert "Verify CSSF 24/856 threshold" in combined

    # 4. Real-time elapsed timer badge in top-right corner of reasoning box
    box_html = get_display_html("- Step 1", "", "reasoning", elapsed_s=2.4, start_ms=1700000000000)
    assert 'data-start-ms="1700000000000"' in box_html
    assert 'data-elapsed="2.4s"' in box_html
    assert '<span class="reasoning-timer">2.4s</span>' in box_html


def test_backend_config_and_dynamic_frontend_interface(monkeypatch, tmp_path):
    from rag_engine.backends.page_index.page_index_config import (
        PAGE_INDEX_CONFIG,
        PageIndexConfig,
        discover_corpus_registry as pi_discover,
    )
    from rag_engine.backends.vector_chunking.vector_chunking_config import (
        VECTOR_CHUNKING_CONFIG,
        VectorChunkingConfig,
        discover_corpus_registry as vc_discover,
    )
    from rag_engine.core.registry import get_backend
    from rag_engine.core.schemas import RetrievalResult, RetrievedChunk
    import rag_agent.tools as tools_mod

    # 1. Verify typed config singletons and generalized structural weights for both RAG modes
    assert isinstance(PAGE_INDEX_CONFIG, PageIndexConfig)
    assert PAGE_INDEX_CONFIG.backend_name == "page_index"
    assert PAGE_INDEX_CONFIG.min_score == 15.0
    assert PAGE_INDEX_CONFIG.requires_warmup is False
    assert PAGE_INDEX_CONFIG.weight_target_doc_match == 6.0
    assert PAGE_INDEX_CONFIG.weight_header_clause_match == 8.0
    assert PAGE_INDEX_CONFIG.domain_concept_boost_scale == 1.0

    assert isinstance(VECTOR_CHUNKING_CONFIG, VectorChunkingConfig)
    assert VECTOR_CHUNKING_CONFIG.backend_name == "vector_chunking"
    assert VECTOR_CHUNKING_CONFIG.min_score == 0.75
    assert VECTOR_CHUNKING_CONFIG.requires_warmup is True
    assert VECTOR_CHUNKING_CONFIG.weight_header_table_match == 1.5
    assert VECTOR_CHUNKING_CONFIG.weight_header_clause_match == 1.5
    assert VECTOR_CHUNKING_CONFIG.domain_concept_boost_scale == 1.0

    # 1b. Verify scalable auto-discovery for arbitrary new documents
    (tmp_path / "esma_guidelines_2026.pdf").write_bytes(b"%PDF-1.4")
    for discover_fn in (pi_discover, vc_discover):
        reg = discover_fn(primary_dir=tmp_path, fallback_dir=tmp_path)
        assert "esma_guidelines_2026.pdf" in reg
        assert reg["esma_guidelines_2026.pdf"].doc_id == "ESMA_GUIDELINES_2026"
        assert reg["esma_guidelines_2026.pdf"].valid_from == "1900-01-01"
        assert reg["esma_guidelines_2026.pdf"].valid_to == "9999-12-31"

    # 2. Verify backend instances expose dynamic interface properties
    pi_backend = get_backend("page_index")
    assert pi_backend.name == "page_index"
    assert pi_backend.min_score == 15.0
    assert pi_backend.requires_warmup is False
    assert pi_backend.warm_up() is False

    vc_backend = get_backend("vector_chunking")
    assert vc_backend.name == "vector_chunking"
    assert vc_backend.min_score == 0.75
    assert vc_backend.requires_warmup is True

    # 3. Verify frontend tools.py reads min_score and warm_up() dynamically from backend
    class _CustomThresholdBackend:
        name = "custom_backend"
        min_score = 42.0
        requires_warmup = False
        warmed_up = False

        def warm_up(self) -> bool:
            self.warmed_up = True
            return False

        def retrieve(self, query: str, audit_date: str | None = None, top_k: int = 5) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                audit_date=audit_date,
                chunks=[
                    RetrievedChunk(
                        id="c1",
                        text="Sample regulatory provision",
                        score=10.0,
                        doc_id="CSSF_24_856",
                        source_file="CSSF24_856eng.pdf",
                        section_id="Section 1",
                        page_number=1,
                        valid_from="2025-01-01",
                        valid_to="9999-12-31",
                    )
                ],
            )

    custom_backend = _CustomThresholdBackend()
    monkeypatch.setattr(tools_mod, "_get_backend", lambda: custom_backend)

    assert tools_mod.warm_up_backend() is False
    assert custom_backend.warmed_up is True

    out = tools_mod.search_documents.invoke(
        {"query": "What is the tolerance threshold?", "audit_date": "2025-02-15"}
    )
    assert "LOW_RELEVANCE_WARNING" in out
    assert "(42.0)" in out


def test_stream_handler_integrity_and_seed_messages(monkeypatch):
    """Verify that _seed_messages and _chat_stream operate without NameError or missing symbols."""
    from db import chat_store
    from langchain_core.messages import AIMessage, HumanMessage
    from ui.stream_handler import _chat_stream, _seed_messages

    conv_id = chat_store.create_conversation("Stream Handler Integ Test")
    try:
        # Seed test history
        chat_store.add_message(conv_id, "user", "Hello auditor", plain="Hello auditor")
        chat_store.add_message(conv_id, "assistant", "<div>Hello!</div>", plain="Hello!")
        chat_store.add_message(conv_id, "user", "What is the threshold?", plain="What is the threshold?")

        # _seed_messages should include prior turns up to the second-to-last turn
        prior = _seed_messages(conv_id)
        assert len(prior) == 2
        assert isinstance(prior[0], HumanMessage) and prior[0].content == "Hello auditor"
        assert isinstance(prior[1], AIMessage) and prior[1].content == "Hello!"

        # _chat_stream should initialize and emit the initial analyzing spinner without NameError
        stream_gen = _chat_stream("Follow up question", [], conv_id, source_mode="RAG only")
        first_chunk = next(stream_gen)
        assert isinstance(first_chunk, str)
        assert "Analyzing" in first_chunk or "status" in first_chunk
    finally:
        try:
            chat_store.delete_conversation(conv_id)
        except Exception:
            pass







