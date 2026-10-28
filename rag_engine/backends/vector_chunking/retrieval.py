"""Bitemporal Hybrid Retrieval & CrossEncoder Reranking Engine (`vector_chunking` backend)."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Literal

from rag_engine.core.retrieval_types import (
    RetrievalResponse,
    RetrievedParentChunk,
    decompose_compound_query,
    infer_target_doc_id,
    parse_date_to_int,
)

from .nomic_baseline import NomicTextEmbeddings, get_or_create_nomic_store
from .vector_chunking_config import (
    DEFAULT_CANDIDATE_K, DEFAULT_RETRIEVAL_MODE, DEFAULT_RETRIEVAL_TOP_K,
    DENSE_DEFAULT_BUCKET_LEN, DENSE_TOKEN_BUCKETS, DOMAIN_CONCEPT_BOOST_SCALE, DOMAIN_CONCEPT_RULES,
    MANIFEST_FILENAME, MAX_QUERY_CACHE_SIZE, MAX_RERANK_CANDIDATES, NOMIC_MODEL_NAME,
    NON_ATOMIC_CHUNK_PRIOR_BONUS, RERANKER_BATCH_SIZE, RERANKER_DOC_MAX_CHARS, RERANKER_MAX_LENGTH,
    RERANKER_MODEL_NAME, RERANKER_QUERY_MAX_CHARS, RERANKER_SHORT_BUCKET_CHAR_THRESHOLD,
    RERANKER_TOKEN_BUCKETS, RUNTIME_ENV_DEFAULTS, SCOPED_DOC_CANDIDATE_LIMIT, SPARSE_VECTOR_NAME,
    SUB_QUERY_STAGE1_BONUS, WEIGHT_BODY_CLAUSE_MATCH, WEIGHT_BODY_TABLE_MATCH,
    WEIGHT_HEADER_CLAUSE_MATCH, WEIGHT_HEADER_TABLE_MATCH, WEIGHT_NUMBERED_POINT_MATCH,
)

for _env_key, _env_val in RUNTIME_ENV_DEFAULTS.items():
    os.environ[_env_key] = _env_val

import torch
from langchain_core.documents import Document
from qdrant_client.http import models as qmodels
from sentence_transformers import CrossEncoder

from .ingest import ComplianceCorpusIngestor, get_optimal_device

logger = logging.getLogger(__name__)
RetrievalModeType = Literal[
    "nomic_dense_only", "dense_only", "dense_bitemporal", "hybrid_no_temporal", "hybrid_bm25", "hybrid_rerank"
]


def build_bitemporal_filter(audit_date: str | None, doc_id_filter: str | None = None) -> qmodels.Filter | None:
    """Construct a Hard Bitemporal Pre-Filter (`valid_from <= audit_date <= valid_to`) for Qdrant."""
    must: list[qmodels.FieldCondition] = []
    if audit_date:
        d_int = parse_date_to_int(audit_date)
        must.append(qmodels.FieldCondition(key="metadata.valid_from_int", range=qmodels.Range(lte=d_int)))
        must.append(qmodels.FieldCondition(key="metadata.valid_to_int", range=qmodels.Range(gte=d_int)))
    if doc_id_filter:
        must.append(qmodels.FieldCondition(key="metadata.doc_id", match=qmodels.MatchValue(value=doc_id_filter)))
    return qmodels.Filter(must=must) if must else None


def compute_domain_alignment_boost(query: str, doc: Document) -> float:
    """Compute a deterministic lexical/structural alignment bonus on top of CrossEncoder logits."""
    q_low = query.lower()
    content_low = doc.page_content.lower()
    section_low = str(doc.metadata.get("section_id", "")).lower()
    boost = 0.0
    for tbl in re.findall(r"\btable\s+\d+\b", q_low):
        is_hdr = f"**{tbl}**" in content_low or tbl in section_low or f"\n{tbl} " in content_low or f"\n{tbl}\n" in content_low
        boost += WEIGHT_HEADER_TABLE_MATCH if is_hdr else (WEIGHT_BODY_TABLE_MATCH if tbl in content_low else 0.0)
    for clause in re.findall(r"\b(?:article\s+\d+|chapter\s+\d+|section\s+(?:[ivx]+|\d+)(?:\.\d+)*(?:\([a-z]\))?)|(?<!\.)\b\d{1,3}\([a-z]\)", q_low):
        in_hdr = clause in section_low or f"### {clause}" in content_low or (clause.endswith(")") and clause[:-3] in section_low and f"-{clause[-3:]}" in section_low)
        boost += WEIGHT_HEADER_CLAUSE_MATCH if in_hdr else (WEIGHT_BODY_CLAUSE_MATCH if clause in content_low else 0.0)
    for pt_num in re.findall(r"\b(?:point|points|paragraph|paragraphs)\s+(\d{1,3})\b", q_low):
        if re.search(rf"(?:^|\n)\s*{pt_num}\.\s", content_low) or f"paragraph {pt_num}" in section_low:
            boost += WEIGHT_NUMBERED_POINT_MATCH
    for pt_a, pt_b in re.findall(r"\b(?:points|paragraphs)\s+(\d{1,3})\s*(?:to|and|–|-)\s*(\d{1,3})\b", q_low):
        if re.search(rf"(?:^|\n)\s*(?:{pt_a}|{pt_b})\.\s", content_low):
            boost += WEIGHT_NUMBERED_POINT_MATCH
    if DOMAIN_CONCEPT_BOOST_SCALE > 0.0:
        for rule in DOMAIN_CONCEPT_RULES:
            if rule.matches(q_low, section_low, content_low):
                boost += rule.weight * DOMAIN_CONCEPT_BOOST_SCALE
    return boost


def _rrf_prefetch(dv: list[float], sv: Any, flt: qmodels.Filter | None, limit: int) -> list[qmodels.Prefetch]:
    sv_q = qmodels.SparseVector(indices=sv.indices, values=sv.values)
    return [
        qmodels.Prefetch(query=dv, using=None, filter=flt, limit=limit),
        qmodels.Prefetch(query=sv_q, using=SPARSE_VECTOR_NAME, filter=flt, limit=limit),
    ]


def _rrf_query_req(dv: list[float], sv: Any, flt: qmodels.Filter | None, limit: int) -> qmodels.QueryRequest:
    return qmodels.QueryRequest(
        prefetch=_rrf_prefetch(dv, sv, flt, limit),
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        filter=flt, limit=limit, with_payload=True, with_vector=False,
    )


class BitemporalComplianceRetriever:
    """Production-grade Bitemporal Hybrid Retriever & CrossEncoder Reranker."""

    def __init__(self, ingestor: ComplianceCorpusIngestor | None = None) -> None:
        self.ingestor = ingestor or ComplianceCorpusIngestor()
        self.ingestor.ingest(clean=False)
        self._reranker: CrossEncoder | None = None
        self._reranker_lock = threading.RLock()
        self._embed_lock = threading.RLock()
        self._nomic_lock = threading.RLock()
        self._nomic_embeddings: NomicTextEmbeddings | None = None
        self._nomic_store: Any = None
        self._dense_query_cache: dict[str, list[float]] = {}
        self._sparse_query_cache: dict[str, Any] = {}
        self._dense_warmed_up = False
        self._reranker_warmed_up = False
        self._parent_cache: dict[str, dict[str, Any]] = {}
        for p_file in self.ingestor.parent_store_dir.glob("*.json"):
            if p_file.name != MANIFEST_FILENAME:
                try:
                    self._parent_cache[p_file.stem] = json.loads(p_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    raise ValueError(f"Corrupt parent chunk JSON {p_file}: {exc}") from exc
        _ = self._embed_dense_query("__warmup__")
        self._dense_query_cache.pop("__warmup__", None)
        _ = self._score_reranker_pairs([["__warmup__", "__warmup__"]])

    @property
    def nomic_embeddings(self) -> NomicTextEmbeddings:
        with self._nomic_lock:
            if self._nomic_embeddings is None:
                self._nomic_embeddings = NomicTextEmbeddings(NOMIC_MODEL_NAME)
            return self._nomic_embeddings

    @property
    def nomic_store(self) -> Any:
        with self._nomic_lock:
            if self._nomic_store is None:
                self._nomic_store = get_or_create_nomic_store(self.ingestor, self.nomic_embeddings)
            return self._nomic_store

    @property
    def reranker(self) -> CrossEncoder:
        with self._reranker_lock:
            if self._reranker is None:
                device = get_optimal_device("reranker")
                dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
                self._reranker = CrossEncoder(
                    RERANKER_MODEL_NAME, device=device, max_length=RERANKER_MAX_LENGTH, model_kwargs={"torch_dtype": dtype}
                )
            return self._reranker

    def _embed_dense_query(self, text: str) -> list[float]:
        with self._embed_lock:
            if (cached := self._dense_query_cache.get(text)) is not None:
                return cached
            st_model = getattr(self.ingestor.dense_embeddings, "_client", None)
            if st_model is None:
                vec = self.ingestor.dense_embeddings.embed_query(text)
            else:
                if not self._dense_warmed_up:
                    self._dense_warmed_up = True
                    with torch.inference_mode():
                        for b in DENSE_TOKEN_BUCKETS:
                            enc = st_model.tokenizer(["warmup " * (b // 2)], padding="max_length", truncation=True, max_length=b, return_tensors="pt")
                            _ = st_model.forward({k: v.to(st_model.device) for k, v in enc.items()})["sentence_embedding"]
                        if hasattr(torch, "mps") and torch.backends.mps.is_available():
                            torch.mps.synchronize()
                clean = text.replace("\n", " ")
                b_len = next((b for b in DENSE_TOKEN_BUCKETS if len(clean) // 3 + 8 <= b), DENSE_DEFAULT_BUCKET_LEN)
                enc = st_model.tokenizer([clean], padding="max_length", truncation=True, max_length=b_len, return_tensors="pt")
                with torch.inference_mode():
                    emb = st_model.forward({k: v.to(st_model.device) for k, v in enc.items()})["sentence_embedding"]
                    vec = torch.nn.functional.normalize(emb, p=2, dim=1)[0].float().cpu().tolist()
            if len(self._dense_query_cache) >= MAX_QUERY_CACHE_SIZE:
                self._dense_query_cache.pop(next(iter(self._dense_query_cache)), None)
            self._dense_query_cache[text] = vec
            return vec

    def _embed_sparse_queries(self, texts: list[str]) -> list[Any]:
        with self._embed_lock:
            missing = [t for t in texts if t not in self._sparse_query_cache]
            if missing:
                new_vecs = (
                    [self.ingestor.sparse_embeddings.embed_query(missing[0])]
                    if len(missing) == 1
                    else list(self.ingestor.sparse_embeddings.embed_documents(missing))
                )
                for t, sv in zip(missing, new_vecs):
                    if len(self._sparse_query_cache) >= MAX_QUERY_CACHE_SIZE:
                        self._sparse_query_cache.pop(next(iter(self._sparse_query_cache)), None)
                    self._sparse_query_cache[t] = sv
            return [self._sparse_query_cache[t] for t in texts]

    def _score_reranker_pairs(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []
        ce_obj, batch_sz, n = self.reranker, RERANKER_BATCH_SIZE, len(pairs)
        if not self._reranker_warmed_up:
            self._reranker_warmed_up = True
            with torch.inference_mode():
                for b_len in RERANKER_TOKEN_BUCKETS:
                    feats = ce_obj.tokenizer([["warmup query", "warmup document"]] * batch_sz, padding="max_length", truncation="longest_first", max_length=b_len, return_tensors="pt")
                    _ = ce_obj.model(**{k: v.to(ce_obj.model.device) for k, v in feats.items()}, return_dict=True).logits.view(-1)
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.synchronize()
        max_chars = max(len(p[0]) + len(p[1]) for p in pairs)
        b_len = RERANKER_TOKEN_BUCKETS[0] if max_chars < RERANKER_SHORT_BUCKET_CHAR_THRESHOLD else RERANKER_TOKEN_BUCKETS[-1]
        pad_n = ((n + batch_sz - 1) // batch_sz) * batch_sz
        padded = pairs + [pairs[-1]] * (pad_n - n)
        out_scores: list[float] = []
        with torch.inference_mode():
            for i in range(0, pad_n, batch_sz):
                feats = ce_obj.tokenizer(padded[i : i + batch_sz], padding="max_length", truncation="longest_first", max_length=b_len, return_tensors="pt")
                logits = ce_obj.model(**{k: v.to(ce_obj.model.device) for k, v in feats.items()}, return_dict=True).logits.view(-1)
                out_scores.extend(ce_obj.activation_fn(logits).float().cpu().tolist())
        return out_scores[:n]

    def load_parent_chunk(self, parent_id: str, score: float = 0.0) -> RetrievedParentChunk | None:
        if (data := self._parent_cache.get(parent_id)) is None:
            path = self.ingestor.parent_store_dir / f"{parent_id}.json"
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(f"Corrupt parent chunk JSON {path}: {exc}") from exc
            self._parent_cache[parent_id] = data
        meta = data.get("metadata", {})
        return RetrievedParentChunk(
            parent_id, meta.get("doc_id", "UNKNOWN"), meta.get("source_file", meta.get("source", "unknown")),
            meta.get("valid_from", ""), meta.get("valid_to", ""), int(meta.get("page_number", 0)),
            meta.get("section_id", ""), data.get("page_content", ""), float(score),
            meta.get("outbound_citations", []), meta.get("sha256", ""), meta.get("md5", ""), meta,
        )

    def retrieve(
        self, query: str, audit_date: str | None = None, mode: RetrievalModeType = DEFAULT_RETRIEVAL_MODE,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K, candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> RetrievalResponse:
        """Execute retrieval in one of the 6 benchmark ablation modes."""
        bitemporal_filter = build_bitemporal_filter(audit_date=audit_date)
        col_name = self.ingestor.collection_name
        if mode in ("nomic_dense_only", "dense_only", "dense_bitemporal", "hybrid_no_temporal", "hybrid_bm25"):
            active_filter = bitemporal_filter if mode in ("dense_bitemporal", "hybrid_bm25") else None
            if mode == "nomic_dense_only":
                raw_docs = self.nomic_store.similarity_search_with_score(query, k=top_k, filter=active_filter)
            else:
                dv = self._embed_dense_query(query)
                if mode in ("dense_only", "dense_bitemporal"):
                    pts = self.ingestor.client.query_points(
                        collection_name=col_name, query=dv, using=None, query_filter=active_filter, limit=top_k, with_payload=True, with_vectors=False
                    ).points
                else:
                    sv = self._embed_sparse_queries([query])[0]
                    pts = self.ingestor.client.query_points(
                        collection_name=col_name, prefetch=_rrf_prefetch(dv, sv, active_filter, top_k),
                        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF), query_filter=active_filter, limit=top_k, with_payload=True, with_vectors=False,
                    ).points
                raw_docs = [
                    (Document(page_content=(pt.payload or {}).get("page_content", ""), metadata=(pt.payload or {}).get("metadata", {})), float(pt.score or 0.0))
                    for pt in pts
                ]
            parent_chunks: list[RetrievedParentChunk] = []
            seen_pids: set[str] = set()
            for doc, score in raw_docs:
                pid = doc.metadata.get("parent_id")
                if pid and pid not in seen_pids:
                    seen_pids.add(pid)
                    if p_chunk := self.load_parent_chunk(pid, score=float(score)):
                        parent_chunks.append(p_chunk)
                        if len(parent_chunks) >= top_k:
                            break
            return RetrievalResponse(query=query, audit_date=audit_date, parent_chunks=parent_chunks, mode=mode)

        sub_queries = decompose_compound_query(query)
        all_queries = [query] + [sq for sq in sub_queries if sq != query]
        dv0 = self._embed_dense_query(query)
        s_vecs = self._embed_sparse_queries(all_queries)
        batch_reqs = [_rrf_query_req(dv0, sv, bitemporal_filter, candidate_k) for sv in s_vecs]
        if target_doc_id := infer_target_doc_id(query):
            batch_reqs.append(_rrf_query_req(dv0, s_vecs[0], build_bitemporal_filter(audit_date, target_doc_id), SCOPED_DOC_CANDIDATE_LIMIT))

        responses = self.ingestor.client.query_batch_points(collection_name=col_name, requests=batch_reqs)
        candidate_map: dict[str, Document] = {}
        cid_rrf_score: dict[str, float] = {}
        sub_query_pids: list[set[str]] = [set() for _ in sub_queries]
        for req_i, resp in enumerate(responses):
            for pt in resp.points:
                payload = pt.payload or {}
                doc = Document(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {}))
                pid = doc.metadata.get("parent_id")
                cid = doc.metadata.get("child_id") or f"{pid}_{hash(doc.page_content)}"
                candidate_map.setdefault(cid, doc)
                cid_rrf_score[cid] = max(cid_rrf_score.get(cid, 0.0), float(pt.score or 0.0))
                if len(sub_queries) > 1 and 1 <= req_i <= len(sub_queries) and pid:
                    sub_query_pids[req_i - 1].add(pid)

        boost_cache: dict[tuple[str, str], float] = {}
        by_parent: dict[str, list[tuple[float, str, Document]]] = {}
        for cid, doc in candidate_map.items():
            if not (pid := doc.metadata.get("parent_id")):
                continue
            for q_s in all_queries:
                boost_cache[(q_s, cid)] = compute_domain_alignment_boost(q_s, doc)
            prior = 0.0 if doc.metadata.get("is_atomic_table_chunk") else NON_ATOMIC_CHUNK_PRIOR_BONUS
            boost = max(boost_cache[(q_s, cid)] for q_s in all_queries) + prior
            by_parent.setdefault(pid, []).append((cid_rrf_score.get(cid, 0.0) + boost, cid, doc))
        rep_items = sorted((max(items, key=lambda x: x[0]) for items in by_parent.values()), key=lambda x: x[0], reverse=True)[:MAX_RERANK_CANDIDATES]
        if not rep_items:
            return RetrievalResponse(query=query, audit_date=audit_date, parent_chunks=[], mode=mode, sub_queries=sub_queries)
        with self._reranker_lock:
            q_ce = query[-RERANKER_QUERY_MAX_CHARS:] if len(query) > RERANKER_QUERY_MAX_CHARS else query
            main_scores = self._score_reranker_pairs([[q_ce, doc.page_content[:RERANKER_DOC_MAX_CHARS]] for _, _, doc in rep_items])

        parent_best_global: dict[str, float] = {}
        parent_best_per_sub: list[dict[str, float]] = [{} for _ in sub_queries]
        for idx, (_, cid, doc) in enumerate(rep_items):
            if not (pid := doc.metadata.get("parent_id")):
                continue
            parent_best_global[pid] = max(parent_best_global.get(pid, float("-inf")), main_scores[idx] + boost_cache[(query, cid)])
            if len(sub_queries) > 1:
                for sq_i, sq in enumerate(sub_queries):
                    sq_bonus = SUB_QUERY_STAGE1_BONUS if pid in sub_query_pids[sq_i] else 0.0
                    sq_score = main_scores[idx] + boost_cache[(sq, cid)] + sq_bonus
                    parent_best_per_sub[sq_i][pid] = max(parent_best_per_sub[sq_i].get(pid, float("-inf")), sq_score)

        selected_pids: list[tuple[str, float]] = []
        seen_pids_set: set[str] = set()
        if len(sub_queries) > 1:
            for sq_i in range(len(sub_queries)):
                ranked_sub = sorted(parent_best_per_sub[sq_i].items(), key=lambda x: x[1], reverse=True)
                if pid := next((p for p, _ in ranked_sub if p not in seen_pids_set), None):
                    seen_pids_set.add(pid)
                    selected_pids.append((pid, parent_best_global.get(pid, 0.0)))
        for pid, score in sorted(parent_best_global.items(), key=lambda x: x[1], reverse=True):
            if len(selected_pids) < top_k and pid not in seen_pids_set:
                seen_pids_set.add(pid)
                selected_pids.append((pid, score))

        parent_chunks = [p_chunk for pid, score in selected_pids[:top_k] if (p_chunk := self.load_parent_chunk(pid, score=score))]
        return RetrievalResponse(query=query, audit_date=audit_date, parent_chunks=parent_chunks, mode=mode, sub_queries=sub_queries)
