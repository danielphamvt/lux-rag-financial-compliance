"""Neuro-Symbolic Second-Level NAV Compliance & Audit Engine.

Shared core verification engine used by the evaluation suite, retrieval backends,
and the frontend LangGraph agent (`verify_nav_and_fee_compliance`).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from rag_engine.core.corpus_parsing import CORPUS_REGISTRY
from rag_engine.core.nav_audit_parsers import (
    parse_prospectus_fee_from_chunks,
    parse_statutory_threshold_from_chunks,
)
from rag_engine.core.nav_audit_schemas import (
    FundAccountingPackInput,
    NAVAuditResult,
    ProspectusFeeSpec,
    StatutoryThresholdSpec,
)
from rag_engine.core.nav_audit_verifier import verify_fee_accrual, verify_nav_pricing
from rag_engine.core.retrieval_types import (
    AUDIT_CONFIDENCE_SCORE,
    AUDIT_EXCERPT_MAX_CHARS,
    AUDIT_RETRIEVAL_TOP_K,
    LOGS_DIR,
)

load_dotenv()
logger = logging.getLogger(__name__)
DEFAULT_AUDIT_RETRIEVAL_MODE: str = os.getenv("VECTOR_CHUNKING_MODE", "hybrid_rerank")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload atomically via a temporary file in `path.parent` and `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class NAVAuditGraphState(TypedDict, total=False):
    accounting_input: FundAccountingPackInput
    retrieval_response: Any
    statutory_spec: StatutoryThresholdSpec
    prospectus_spec: ProspectusFeeSpec | None
    prospectus_resolved_via: str
    logs_subpath: str | None
    audit_result: NAVAuditResult


class NeuroSymbolicNAVAuditEngine:
    """Executes Second-Level NAV Compliance & Audit Verification via LangGraph StateGraph."""

    def __init__(self, retriever: Any | None = None, logs_dir: Path = LOGS_DIR):
        if retriever is None:
            from rag_engine.core.registry import get_backend

            retriever = get_backend("vector_chunking").retriever
        self.retriever = retriever
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.graph = self._build_graph()

    @staticmethod
    def _statutory_category_hint(fund_type: str) -> str:
        ft_low = fund_type.lower()
        if "money market" in ft_low or "mmf" in ft_low or "cash" in ft_low:
            return " (money market UCIs / MMF Table 1)"
        if "bond" in ft_low:
            return " (bond UCIs Table 2 Table 3)"
        if "mixed" in ft_low:
            return " (mixed UCIs Table 2 Table 3)"
        return " (equity UCIs / shares Table 2)"

    @staticmethod
    def _resolve_prospectus(inp: FundAccountingPackInput) -> tuple[str, str, str]:
        """Resolve the prospectus document for a fee audit: explicit input first, else a unique corpus-registry match.

        Args:
            inp: Fund accounting pack carrying `fund_name` and an optional explicit `prospectus_doc_id`.

        Returns:
            Tuple of (resolved doc_id, resolution provenance "explicit"|"registry", registry title).

        Raises:
            ValueError: On an unknown registry doc_id, a fund name matching no prospectus, or an ambiguous match.
        """

        def _norm(s: str) -> str:
            return " ".join(s.casefold().split())

        specs_by_id = {spec.doc_id: spec for spec in CORPUS_REGISTRY.values()}
        if inp.prospectus_doc_id:
            spec = specs_by_id.get(inp.prospectus_doc_id) or CORPUS_REGISTRY.get(inp.prospectus_doc_id)
            if spec is None:
                raise ValueError(
                    f"prospectus_doc_id {inp.prospectus_doc_id!r} not found in the corpus registry. "
                    f"Registered documents: {', '.join(sorted(specs_by_id))}."
                )
            return spec.doc_id, "explicit", spec.title
        prospectus_specs = [spec for spec in CORPUS_REGISTRY.values() if "prospectus" in _norm(spec.title)]
        name = _norm(inp.fund_name)
        matches = [spec for spec in prospectus_specs if name and name in _norm(spec.title)]
        if len(matches) == 1:
            spec = matches[0]
            return spec.doc_id, "registry", spec.title
        available = "; ".join(f"{spec.title} ({spec.doc_id})" for spec in sorted(prospectus_specs, key=lambda s: s.doc_id))
        if not matches:
            raise ValueError(
                f"Fund name {inp.fund_name!r} does not uniquely match any prospectus in the corpus registry. "
                f"Available prospectus funds: {available}. Pass prospectus_doc_id explicitly for unregistered documents."
            )
        raise ValueError(
            f"Fund name {inp.fund_name!r} matches multiple prospectus documents ({available}); "
            "pass prospectus_doc_id to disambiguate."
        )

    def _retrieve_node(self, state: NAVAuditGraphState) -> dict[str, Any]:
        inp = state["accounting_input"]
        cat_hint = self._statutory_category_hint(inp.fund_type)
        if inp.audit_type == "fee_accrual_audit":
            query = (
                f"What is the management fee rate, tiered fee schedule, and day-count convention "
                f"for {inp.fund_name} {inp.share_class or ''}?"
            )
            resp = self.retriever.retrieve(query, inp.audit_date, DEFAULT_AUDIT_RETRIEVAL_MODE, AUDIT_RETRIEVAL_TOP_K)
            stat_query = (
                f"What is the NAV error tolerance threshold in % of NAV for {inp.fund_type}{cat_hint}, "
                f"and what are the EUR indemnification or compensation thresholds for a separate/special report?"
            )
            stat_resp = self.retriever.retrieve(stat_query, inp.audit_date, DEFAULT_AUDIT_RETRIEVAL_MODE, AUDIT_RETRIEVAL_TOP_K)
            seen = {c.parent_id for c in resp.parent_chunks}
            for sc in stat_resp.parent_chunks:
                if sc.parent_id not in seen:
                    resp.parent_chunks.append(sc)
                    seen.add(sc.parent_id)
            return {"retrieval_response": resp}

        query = (
            f"What is the NAV error tolerance threshold in % of NAV for {inp.fund_type}{cat_hint}, "
            f"and what are the EUR indemnification or compensation thresholds for a separate/special report?"
        )
        return {"retrieval_response": self.retriever.retrieve(query, inp.audit_date, DEFAULT_AUDIT_RETRIEVAL_MODE, AUDIT_RETRIEVAL_TOP_K)}

    def _semantic_parse_node(self, state: NAVAuditGraphState) -> dict[str, Any]:
        inp, retrieval = state["accounting_input"], state["retrieval_response"]
        statutory_spec = parse_statutory_threshold_from_chunks(retrieval, inp.fund_type)
        prospectus_spec, resolved_via = None, ""
        if inp.audit_type == "fee_accrual_audit":
            doc_id, resolved_via, registry_title = self._resolve_prospectus(inp)
            prospectus_spec = parse_prospectus_fee_from_chunks(
                retrieval,
                share_class=inp.share_class or "Class I",
                prospectus_doc_id=doc_id,
                fund_name=inp.fund_name,
                registry_title=registry_title,
            )
        return {"statutory_spec": statutory_spec, "prospectus_spec": prospectus_spec, "prospectus_resolved_via": resolved_via}


    def _deterministic_verify_node(self, state: NAVAuditGraphState) -> dict[str, Any]:
        inp, retrieval, stat, prosp = state["accounting_input"], state["retrieval_response"], state["statutory_spec"], state.get("prospectus_spec")
        if inp.audit_type == "fee_accrual_audit" and prosp is not None:
            outcome = verify_fee_accrual(inp, stat, prosp)
        else:
            outcome = verify_nav_pricing(inp, stat)
        steps, control_flags, verdict, rationale, variance_bps, is_material, special_audit_req, reported_val_str, replicated_val_str, abs_var_str = outcome

        chunks_meta = [
            {
                "parent_id": c.parent_id, "doc_id": c.doc_id, "source_file": c.source_file, "valid_from": c.valid_from,
                "valid_to": c.valid_to, "page_number": c.page_number, "section_id": c.section_id,
                "rerank_score": round(c.rerank_score, 4), "sha256": c.sha256, "md5": c.md5,
                "outbound_citations": c.outbound_citations,
                "excerpt": c.page_content[:AUDIT_EXCERPT_MAX_CHARS] + ("..." if len(c.page_content) > AUDIT_EXCERPT_MAX_CHARS else ""),
            }
            for c in retrieval.parent_chunks
        ]
        extracted_params: dict[str, Any] = {"statutory_threshold_spec": json.loads(stat.model_dump_json())}
        if prosp is not None:
            extracted_params["prospectus_fee_spec"] = json.loads(prosp.model_dump_json())

        return {
            "audit_result": NAVAuditResult(
                run_id=f"run_{inp.scenario_id}_{uuid.uuid4().hex[:8]}", scenario_id=inp.scenario_id,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), audit_date=inp.audit_date,
                fund_name=inp.fund_name, fund_type=inp.fund_type, share_class=inp.share_class,
                retrieved_chunks_metadata=chunks_meta, extracted_pydantic_parameters=extracted_params,
                python_decimal_calculation_steps=steps, reported_value=reported_val_str,
                replicated_value=replicated_val_str, absolute_variance=abs_var_str, variance_bps=variance_bps,
                applicable_threshold_bps=stat.tolerance_threshold_bps, is_material_nav_breach=is_material,
                special_audit_report_required=special_audit_req, confidence_score=AUDIT_CONFIDENCE_SCORE,
                control_flags=control_flags, regulatory_verdict=verdict, verdict_rationale=rationale,
                prospectus_resolved_via=str(state.get("prospectus_resolved_via") or ""),
            )
        }

    def _audit_log_writer_node(self, state: NAVAuditGraphState) -> dict[str, Any]:
        result = state["audit_result"]
        subpath = state.get("logs_subpath") or ("live_chat" if result.scenario_id == "live_chat" else None)
        log_path = (
            self.logs_dir / subpath / f"audit_trail_{result.run_id}.json"
            if subpath
            else self.logs_dir / f"audit_trail_{result.scenario_id}.json"
        )
        payload_without_hash = json.loads(result.model_dump_json())
        payload_bytes = json.dumps(payload_without_hash, sort_keys=True, ensure_ascii=False).encode("utf-8")
        result.audit_trail_path = str(log_path)
        result.audit_trail_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        _write_json_atomic(log_path, json.loads(result.model_dump_json()))
        return {"audit_result": result}

    def _build_graph(self) -> Any:
        builder = StateGraph(NAVAuditGraphState)
        builder.add_node("bitemporal_retrieve", self._retrieve_node)
        builder.add_node("semantic_rule_parser", self._semantic_parse_node)
        builder.add_node("deterministic_decimal_verifier", self._deterministic_verify_node)
        builder.add_node("immutable_audit_trail_writer", self._audit_log_writer_node)
        builder.add_edge(START, "bitemporal_retrieve")
        builder.add_edge("bitemporal_retrieve", "semantic_rule_parser")
        builder.add_edge("semantic_rule_parser", "deterministic_decimal_verifier")
        builder.add_edge("deterministic_decimal_verifier", "immutable_audit_trail_writer")
        builder.add_edge("immutable_audit_trail_writer", END)
        return builder.compile()

    def execute_audit(self, accounting_input: FundAccountingPackInput, logs_subpath: str | None = None) -> NAVAuditResult:
        """Execute the audit graph for one accounting pack.

        Args:
            accounting_input: Fund accounting pack to audit.
            logs_subpath: Optional explicit subdirectory under `logs_dir` for the audit trail; when None,
                `live_chat/` is used iff `scenario_id == "live_chat"` (flat filename otherwise).

        Returns:
            The immutable `NAVAuditResult` with verdict, steps, and audit-trail integrity fields.
        """
        return self.graph.invoke({"accounting_input": accounting_input, "logs_subpath": logs_subpath})["audit_result"]
