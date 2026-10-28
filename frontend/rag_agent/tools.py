"""Frontend retrieval tool: thin wrapper over the shared backend interface.

Derives conversation-level parameters (`audit_date`, `source`) from `PROFILE` and formats results.
"""

import logging
import re
import threading
import time
from decimal import Decimal
from pathlib import Path

from langchain_core.tools import tool

from config import DEFAULT_SEARCH_LIMIT, SOURCE_NAMES
from domain_profile import PROFILE
from rag_agent.out_of_corpus import _detect_out_of_corpus_instruments
from rag_agent.harness import evaluate_decimal_expressions, format_nav_audit_output, validate_nav_audit_contract
from rag_engine.core.nav_audit import FundAccountingPackInput, NeuroSymbolicNAVAuditEngine
from rag_engine.core.nav_audit_schemas import round_bps, round_currency
from rag_engine.core.registry import get_backend

logger = logging.getLogger(__name__)
_backend = None
_backend_lock = threading.Lock()


def _get_backend():
    """Resolve the backend once (RAG_BACKEND env or registry default); thread-safe —
    LangGraph may invoke search_documents from parallel sub-query branches."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = get_backend()
    return _backend


def warm_up_backend() -> bool:
    """Warm up the active backend via its `ComplianceBackend` interface contract."""
    backend = _get_backend()
    warmed = bool(backend.warm_up())
    if not warmed:
        logger.info("Backend '%s' does not require model warm-up. Skipping.", backend.name)
    return warmed


def _regime_cfg() -> dict:
    return PROFILE.get("bitemporal_regimes", {})


def _matches_regime_branch(q_low: str, branch: dict) -> bool:
    years, tokens = branch.get("years", []), branch.get("tokens", [])
    year_hit = bool(years and re.search(r"\b(?:" + "|".join(re.escape(y) for y in years) + r")\b", q_low))
    return year_hit or any(tok.lower() in q_low for tok in tokens)


def _spans_both_regimes(query: str) -> bool:
    """Return True when the query references both predecessor and successor regimes in `PROFILE`."""
    q_low = (query or "").lower()
    cfg = _regime_cfg()
    pred, succ, trans = cfg.get("predecessor", {}), cfg.get("successor", {}), cfg.get("transitional", {})
    trans_kws = [k.lower() for k in trans.get("keywords", [])]
    trans_excl = [e.lower() for e in trans.get("exclude_when_tokens", [])]
    pred_tokens = [t.lower() for t in pred.get("tokens", [])]
    if any(k in q_low for k in trans_kws) and not any(e in q_low for e in trans_excl) and not any(t in q_low for t in pred_tokens):
        return False
    return _matches_regime_branch(q_low, pred) and _matches_regime_branch(q_low, succ)


def _infer_audit_date_from_query(query: str) -> str | None:
    """Infer YYYY-MM-DD audit_date from natural-language query using `PROFILE['bitemporal_regimes']`."""
    if _spans_both_regimes(query):
        return None
    if iso_match := re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", query or ""):
        return iso_match.group(1)
    q_low, cfg = (query or "").lower(), _regime_cfg()
    trans = cfg.get("transitional", {})
    if any(k.lower() in q_low for k in trans.get("keywords", [])) and not any(e.lower() in q_low for e in trans.get("exclude_when_tokens", [])):
        return trans.get("default_audit_date")
    pred, succ = cfg.get("predecessor", {}), cfg.get("successor", {})
    has_pred, has_succ = _matches_regime_branch(q_low, pred), _matches_regime_branch(q_low, succ)
    return pred.get("default_audit_date") if (has_pred and not has_succ) else (succ.get("default_audit_date") if (has_succ and not has_pred) else None)


def _split_query_clauses(query: str) -> list[str]:
    """Split multi-clause questions along general grammatical boundaries and multi-instrument boundaries."""
    q_norm = (query or "").strip()
    parts = re.split(r";\s*|(?<!\w)\(\d+\)\s+|\bversus\b", q_norm, flags=re.IGNORECASE)
    inst_pat, src_kws = _regime_cfg().get("instrument_split_pattern", ""), [k.lower() for k in PROFILE.get("source_keywords", [])]
    if inst_pat and _spans_both_regimes(q_norm) and len(src_kws) >= 2:
        parts = [sp for p in parts for sp in (re.split(inst_pat, p, flags=re.I) if sum(1 for k in src_kws if k.replace("_", "/") in p.lower() or k.replace("_", " ") in p.lower()) >= 2 else [p])]
    clauses = [p.strip(" ,?.") for p in parts if len(p.strip(" ,?.")) > 18]
    return [q_norm, *clauses] if len(clauses) > 1 else [q_norm]


def _format_chunk_location(section_id: str | None, page_number: int | None) -> str:
    sec = (section_id or "").strip()
    if page_number is not None and sec and f"page {page_number}" not in sec.lower():
        return f"{sec} (Page {page_number})"
    if sec:
        return sec
    return f"Page {page_number}" if page_number is not None else ""


def _format_filter_regime(dates: tuple, audit_date: str | None, source: str | None) -> str:
    cfg = _regime_cfg()
    pred_lbl = cfg.get("predecessor", {}).get("label", "Predecessor")
    succ_lbl = cfg.get("successor", {}).get("label", "Successor")
    cutoff = cfg.get("cutoff_date", "2025-01-01")
    if len(dates) > 1:
        regime = f"audit_date = {' & '.join(dates)} ({pred_lbl} & {succ_lbl})"
    elif audit_date:
        circ = succ_lbl if audit_date >= cutoff else pred_lbl
        regime = f"audit_date = {audit_date} ({circ})"
    else:
        regime = "audit_date = None (All regimes)"
    if source:
        regime += f" • source = {Path(source).stem}"
    return regime


@tool(response_format="content_and_artifact")
def search_documents(query: str, limit: int = DEFAULT_SEARCH_LIMIT, source: str = None, audit_date: str = None) -> tuple[str, dict | None]:
    """Search the indexed bitemporal compliance corpus and return full parent contexts.

    Args:
        query: Search query string
        limit: Maximum number of parent documents to return (default from config)
        source: Optional exact source file name
        audit_date: Optional point-in-time audit date (YYYY-MM-DD) for bitemporal filtering
    """
    try:
        if source and source not in SOURCE_NAMES:
            source = None
        if audit_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", audit_date.strip()):
            audit_date = _infer_audit_date_from_query(f"{audit_date} {query}")
        spans_both = _spans_both_regimes(query)
        if spans_both:
            audit_date = None
        elif not audit_date:
            audit_date = _infer_audit_date_from_query(query)

        top_k = max(1, int(limit or DEFAULT_SEARCH_LIMIT))
        backend = _get_backend()
        clauses = _split_query_clauses(query)
        cfg = _regime_cfg()
        dual_dates = (
            cfg.get("predecessor", {}).get("default_audit_date", "2024-06-15"),
            cfg.get("successor", {}).get("default_audit_date", "2025-02-15"),
        )
        dates = dual_dates if spans_both else (audit_date,)
        backend_name = backend.name

        t0 = time.perf_counter()
        seen_ids: set[str] = set()
        chunks = []
        for dt in dates:
            primary_k = 3 if len(dates) > 1 else top_k
            for idx, cl in enumerate(clauses):
                k = primary_k if idx == 0 else 2
                for c in backend.retrieve(query=cl, audit_date=dt, top_k=k).chunks:
                    if c.id not in seen_ids:
                        seen_ids.add(c.id)
                        chunks.append(c)
        latency_ms = max(1, round((time.perf_counter() - t0) * 1000))

        if source:
            src_name = Path(source).name
            filtered = [c for c in chunks if c.source_file and Path(c.source_file).name == src_name]
            if filtered:
                chunks = filtered

        doc_labels = PROFILE.get("document_labels", {})
        sub_queries = [cl.strip() for cl in clauses[1:]]
        telemetry = {
            "type": "rag_telemetry",
            "backend": backend_name,
            "latency_ms": latency_ms,
            "query": query,
            "sub_queries": sub_queries,
            "filter_regime": _format_filter_regime(dates, audit_date, source),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "score": round(float(c.score), 2),
                    "doc_label": doc_labels.get(c.doc_id) or (Path(c.source_file).stem if c.source_file else c.doc_id or "Document"),
                    "location": _format_chunk_location(c.section_id, c.page_number),
                }
                for c in sorted(chunks, key=lambda item: item.score, reverse=True)
            ],
        }

        if not chunks:
            return PROFILE["strings"]["retrieval_no_results"], telemetry

        min_score = float(getattr(backend, "min_score", 0.75))
        top_score = max((c.score for c in chunks), default=0.0)
        warning_header = (
            f"[LOW_RELEVANCE_WARNING: Top retrieval score ({top_score:.2f}) is below the calibrated domain threshold ({min_score}). "
            f"Verify whether the queried statute or rule appears in the chunks below or in your other search calls; "
            f"if the entire queried regime is outside the indexed corpus, you MUST abstain (whereas if an in-corpus rule or self-contained mandate test is being audited alongside an unindexed clause, complete the analysis and note the unindexed clause).]\n\n"
            if top_score < min_score else ""
        )
        out_of_corpus = _detect_out_of_corpus_instruments(query, chunks)
        if out_of_corpus:
            corpus_list_str = ", ".join(doc_labels.values()) if doc_labels else "the indexed compliance corpus"
            ooc_header = (
                f"[OUT_OF_CORPUS_INSTRUMENT_WARNING: The query references external regulatory instrument(s) "
                f"({', '.join(out_of_corpus)}) that are NOT part of the indexed compliance corpus ({corpus_list_str}). "
                f"In Strict RAG-only mode, you MUST abstain if the query relies exclusively on these external instruments (if the query also audits in-corpus provisions, complete the full in-corpus and quantitative analysis and state that {', '.join(out_of_corpus)} is not in the indexed corpus); in Assistant mode you MUST "
                f"include the mandatory yellow warning triangle banner (⚠️ **WARNING — Data Source: ...**) immediately below ### CONCLUSION.]\n\n"
            )
            warning_header = ooc_header + warning_header

        contexts = [
            f"Parent ID: {c.id}\n"
            f"Document ID: {c.doc_id or 'unknown'}\n"
            f"File Name: {c.source_file or 'unknown'}\n"
            f"Effective Window: {c.valid_from or ''} to {c.valid_to or ''}\n"
            f"Section / Page: {c.section_id or ''} (Page {c.page_number if c.page_number is not None else ''})\n"
            f"Content: {c.text.strip()}"
            for c in chunks
        ]
        return warning_header + "\n\n".join(contexts), telemetry

    except Exception as e:
        return f"RETRIEVAL_ERROR: {e!s}", None


@tool
def run_nav_audit(
    audit_date: str = "2025-02-15",
    fund_type: str = "Bond UCITS",
    audit_type: str = "nav_pricing_error",
    fund_name: str = "Audited UCI",
    replicated_net_assets_eur: str | float | None = None,
    shares_outstanding: str | float | None = None,
    reported_nav_per_share: str | float | None = None,
    total_compensation_eur: str | float | None = None,
    max_single_investor_eur: str | float | None = None,
    share_class: str | None = None,
    gav_eur: str | float | None = None,
    accrual_days: int | None = None,
    admin_reported_fee_eur: str | float | None = None,
    admin_used_day_count: str | None = None,
    prospectus_doc_id: str | None = None,
) -> str:
    """Execute deterministic Symbolic Decimal (ROUND_HALF_UP) NAV or fee accrual audit verification.

    Args:
        admin_used_day_count: Day-count convention the administrator actually used (e.g. "Actual/360");
            when omitted, the erroneous-convention replication is labeled as assumed.
        prospectus_doc_id: Explicit prospectus document ID overriding corpus-registry fund-name resolution.
    """
    try:
        to_dec = lambda v: Decimal(str(v).replace(",", "").strip()) if v is not None and str(v).strip() != "" else None
        rep_assets_dec = to_dec(replicated_net_assets_eur)
        shares_dec = to_dec(shares_outstanding)
        rep_nav_dec = to_dec(reported_nav_per_share)
        gav_dec = to_dec(gav_eur)

        guard_err = validate_nav_audit_contract(
            audit_type=audit_type,
            share_class=share_class,
            replicated_net_assets_eur=rep_assets_dec,
            shares_outstanding=shares_dec,
            reported_nav_per_share=rep_nav_dec,
            gav_eur=gav_dec,
        )
        if guard_err:
            return guard_err

        cfg = _regime_cfg()
        fallback_date = cfg.get("default_audit_date", "2025-02-15")
        norm_date = (
            audit_date.strip()
            if re.match(r"^\d{4}-\d{2}-\d{2}$", str(audit_date).strip())
            else (_infer_audit_date_from_query(str(audit_date)) or fallback_date)
        )
        eff_type = "fee_accrual_audit" if (audit_type == "fee_accrual_audit" or gav_dec is not None) else "nav_pricing_error"
        has_explicit_admin_fee = to_dec(admin_reported_fee_eur) is not None
        supported_cls = PROFILE.get("supported_share_classes") or []
        eff_cls = share_class if (share_class and share_class in supported_cls) else (supported_cls[0] if supported_cls else share_class)
        default_days = int(cfg.get("default_accrual_days", 365))

        inp = FundAccountingPackInput(
            scenario_id="live_chat",
            audit_date=norm_date,
            audit_type=eff_type,
            fund_name=fund_name,
            fund_type=fund_type or cfg.get("default_fund_type", "Equity UCITS"),
            share_class=eff_cls if eff_type == "fee_accrual_audit" else None,
            replicated_net_assets_eur=rep_assets_dec,
            shares_outstanding=shares_dec,
            reported_nav_per_share=rep_nav_dec,
            total_compensation_eur=to_dec(total_compensation_eur),
            max_single_investor_eur=to_dec(max_single_investor_eur),
            gav_eur=gav_dec,
            accrual_days=int(accrual_days) if accrual_days is not None else (default_days if eff_type == "fee_accrual_audit" else None),
            admin_reported_fee_eur=to_dec(admin_reported_fee_eur),
            admin_used_day_count=str(admin_used_day_count).strip() if admin_used_day_count else None,
            prospectus_doc_id=str(prospectus_doc_id).strip() if prospectus_doc_id else None,
        )
        engine = NeuroSymbolicNAVAuditEngine(retriever=_get_backend().retriever)
        res = engine.execute_audit(inp, logs_subpath="live_chat")
        return format_nav_audit_output(
            res=res,
            eff_type=eff_type,
            eff_cls=eff_cls or "",
            has_explicit_admin_fee=has_explicit_admin_fee,
            round_bps_fn=round_bps,
            round_currency_fn=round_currency,
        )
    except Exception as e:
        return f"AUDIT_ENGINE_ERROR: {e!s}"


@tool
def calculate_decimal(expressions: list[str]) -> str:
    """Evaluate arbitrary financial arithmetic expressions using exact Python Decimal (ROUND_HALF_UP)."""
    return evaluate_decimal_expressions(expressions)
