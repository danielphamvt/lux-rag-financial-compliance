"""SOTA Page-Level & Document-Tree Navigation Retriever (`page_index` backend)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rag_engine.backends.page_index.ingest import PageIndexIngestor
from rag_engine.backends.page_index.page_index_config import (
    BACKEND_NAME,
    BM25_B,
    BM25_K1,
    COMPOUND_QUERY_SEPARATORS,
    DEFAULT_RETRIEVAL_TOP_K,
    DOC_ROUTING_KEYWORDS,
    DOMAIN_CONCEPT_BOOST_SCALE,
    DOMAIN_CONCEPT_RULES,
    MAX_POINT_RANGE_SPAN,
    MIN_TOKEN_LENGTH,
    NON_TEMPORAL_ABLATION_MODES,
    STOPWORDS,
    TOKEN_REGEX,
    UNSUPPORTED_DENSE_MODES,
    WEIGHT_BODY_CLAUSE_MATCH,
    WEIGHT_BODY_TABLE_MATCH,
    WEIGHT_HEADER_CLAUSE_MATCH,
    WEIGHT_HEADER_TABLE_MATCH,
    WEIGHT_NUMBERED_POINT_MATCH,
    WEIGHT_TARGET_DOC_MATCH,
)
from rag_engine.core.retrieval_types import (
    RetrievalResponse,
    RetrievedParentChunk,
    decompose_compound_query as _shared_decompose_compound_query,
    infer_target_doc_id as _shared_infer_target_doc_id,
    validate_audit_date as _validate_audit_date,
)

_TABLE_RE = re.compile(r"\btable\s+\d+\b")
_CLAUSE_RE = re.compile(r"\b(?:article\s+\d+|chapter\s+\d+|section\s+(?:[ivx]+|\d+)(?:\.\d+)*(?:\([a-z]\))?)|(?<!\.)\b\d{1,3}\([a-z]\)")
_POINT_SINGLE_RE = re.compile(r"\b(?:point|points|paragraph|paragraphs)\s+(\d{1,3})\b")
_POINT_RANGE_RE = re.compile(r"\b(?:points|paragraphs)\s+(\d{1,3})\s*(?:to|and|–|-)\s*(\d{1,3})\b")


@dataclass(frozen=True)
class ParsedQueryTargets:
    """Pre-extracted structural query targets for fast per-page tree scoring."""

    q_low: str
    target_doc_id: str | None
    tables: tuple[str, ...]
    clauses: tuple[str, ...]
    target_points: frozenset[int]


def parse_query_targets(query: str, target_doc_id: str | None = None) -> ParsedQueryTargets:
    """Pre-compute lowercase query, table refs, clause refs, and point numbers once per query.

    Args:
        query: Raw user or benchmark query string.
        target_doc_id: Optional inferred target document ID.
    Returns:
        Frozen `ParsedQueryTargets` container.
    """
    q_low = query.lower()
    pts: set[int] = {int(m) for m in _POINT_SINGLE_RE.findall(q_low)}
    for pt_a, pt_b in _POINT_RANGE_RE.findall(q_low):
        a_i, b_i = int(pt_a), int(pt_b)
        pts.update(range(a_i, b_i + 1) if 0 <= b_i - a_i <= MAX_POINT_RANGE_SPAN else (a_i, b_i))
    return ParsedQueryTargets(
        q_low=q_low,
        target_doc_id=target_doc_id,
        tables=tuple(_TABLE_RE.findall(q_low)),
        clauses=tuple(_CLAUSE_RE.findall(q_low)),
        target_points=frozenset(pts),
    )


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(TOKEN_REGEX, text.lower()) if t not in STOPWORDS and len(t) >= MIN_TOKEN_LENGTH]


def decompose_compound_query(query: str) -> list[str]:
    """Decompose compound and multi-target compliance queries into focused sub-queries.

    Args:
        query: Input compliance query.
    Returns:
        Ordered list of deduplicated sub-queries.
    """
    q_clean = query.strip()
    sub_queries = list(_shared_decompose_compound_query(q_clean, separators=COMPOUND_QUERY_SEPARATORS))
    articles = re.findall(r"\bArticle\s+\d+\b", q_clean, flags=re.IGNORECASE)
    if len(articles) >= 2:
        sub_queries.extend(f"{art} {q_clean}" for art in dict.fromkeys(articles))
    tables = re.findall(r"\bTable\s+\d+\b", q_clean, flags=re.IGNORECASE)
    if len(tables) >= 2:
        sub_queries.extend(f"{tbl} {q_clean}" for tbl in dict.fromkeys(tables))
    for pt_a, pt_b in _POINT_RANGE_RE.findall(q_clean.lower()):
        a_int, b_int = int(pt_a), int(pt_b)
        if a_int != b_int:
            sub_queries.extend((f"Point {a_int} {q_clean}", f"Point {b_int} {q_clean}"))
    return list(dict.fromkeys(sub_queries))


def infer_target_doc_id(query: str) -> str | None:
    """Infer target document ID from explicit regulatory identifiers in the query.

    Args:
        query: Input compliance query.
    Returns:
        Matched `doc_id` or `None`.
    """
    return _shared_infer_target_doc_id(query, routing_keywords=DOC_ROUTING_KEYWORDS)


def compute_tree_navigation_score(
    query: str,
    page: dict[str, Any],
    target_doc_id: str | None,
    *,
    targets: ParsedQueryTargets | None = None,
    content_low: str | None = None,
    section_low: str | None = None,
    summary_low: str | None = None,
) -> float:
    """Deterministic hierarchical Document Tree navigation score (Document -> Section -> Page).

    Args:
        query: Query or sub-query string.
        page: Page node dictionary.
        target_doc_id: Inferred target document ID.
        targets: Optional pre-parsed query targets.
        content_low: Optional cached lowercase page content.
        section_low: Optional cached lowercase section ID.
        summary_low: Optional cached lowercase page summary.
    Returns:
        Numeric structural tree-navigation boost score.
    """
    parsed = targets or parse_query_targets(query, target_doc_id)
    q_low = parsed.q_low
    eff_target_doc = parsed.target_doc_id if targets is not None else target_doc_id
    c_low = content_low if content_low is not None else page["page_content"].lower()
    sec_low = section_low if section_low is not None else str(page.get("section_id", "")).lower()
    sum_low = summary_low if summary_low is not None else str(page.get("summary", "")).lower()
    score = WEIGHT_TARGET_DOC_MATCH if (eff_target_doc and page["doc_id"] == eff_target_doc) else 0.0

    for tbl in parsed.tables:
        if f"**{tbl}**" in c_low or re.search(rf"\b{re.escape(tbl)}\b", sec_low):
            score += WEIGHT_HEADER_TABLE_MATCH
        elif re.search(rf"\b{re.escape(tbl)}\b", c_low):
            score += WEIGHT_BODY_TABLE_MATCH
    for clause in parsed.clauses:
        if clause in sec_low or f"### {clause}" in c_low or (clause.endswith(")") and clause[:-3] in sec_low and f"-{clause[-3:]}" in sec_low):
            score += WEIGHT_HEADER_CLAUSE_MATCH
        elif clause in c_low:
            score += WEIGHT_BODY_CLAUSE_MATCH
    for pt in parsed.target_points:
        if re.search(rf"(?:^|\n)\s*{pt}\.\s", c_low) or f"point {pt}" in sum_low:
            score += WEIGHT_NUMBERED_POINT_MATCH
    if DOMAIN_CONCEPT_BOOST_SCALE > 0.0:
        for rule in DOMAIN_CONCEPT_RULES:
            if rule.matches(q_low, sec_low, c_low):
                score += rule.weight * DOMAIN_CONCEPT_BOOST_SCALE
    return score


class PageIndexRetriever:
    """Deterministic Bitemporal Document-Tree & Page-Level BM25 Retriever."""

    nomic_store = None
    nomic_embeddings = None

    def __init__(self, ingestor: PageIndexIngestor | None = None):
        self.ingestor = ingestor or PageIndexIngestor()
        if not self.ingestor.tree_path.exists():
            self.ingestor.ingest(clean=False)
        self._load_tree()

    def _load_tree(self) -> None:
        raw = json.loads(self.ingestor.tree_path.read_text(encoding="utf-8"))
        self.tree = raw.get("documents", {})
        self.all_pages: list[dict[str, Any]] = []
        self.page_by_id: dict[str, dict[str, Any]] = {}
        self._content_low: dict[str, str] = {}
        self._section_low: dict[str, str] = {}
        self._summary_low: dict[str, str] = {}

        for doc_node in self.tree.values():
            for p in doc_node.get("pages", []):
                pid = p["parent_id"]
                self._content_low[pid] = p["page_content"].lower()
                self._section_low[pid] = str(p.get("section_id", "")).lower()
                self._summary_low[pid] = str(p.get("summary", "")).lower()
                self.all_pages.append(p)
                self.page_by_id[pid] = p

        self._doc_freqs: Counter[str] = Counter()
        self._page_tf: dict[str, Counter[str]] = {}
        self._page_len: dict[str, int] = {}
        total_len = 0
        for p in self.all_pages:
            pid = p["parent_id"]
            tokens = _tokenize(f"{p.get('summary', '')} {p['page_content']}")
            tf = Counter(tokens)
            self._page_tf[pid], self._page_len[pid] = tf, len(tokens)
            total_len += len(tokens)
            for term in tf:
                self._doc_freqs[term] += 1
        self._avgdl = total_len / len(self.all_pages) if self.all_pages else 1.0
        self._n_pages = len(self.all_pages)

    def _bm25_score(self, query_tokens: list[str], pid: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf_map = self._page_tf.get(pid)
        if not tf_map or not query_tokens:
            return 0.0
        dl, score = self._page_len.get(pid, 1), 0.0
        for term in query_tokens:
            f = tf_map.get(term, 0)
            if f:
                df = self._doc_freqs.get(term, 0)
                idf = math.log(1.0 + (self._n_pages - df + 0.5) / (df + 0.5))
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (dl / self._avgdl)))
        return score

    @staticmethod
    def _to_parent_chunk(page: dict[str, Any], score: float) -> RetrievedParentChunk:
        return RetrievedParentChunk(
            parent_id=page["parent_id"],
            doc_id=page["doc_id"],
            source_file=page["source_file"],
            valid_from=page["valid_from"],
            valid_to=page["valid_to"],
            page_number=int(page["page_number"]),
            section_id=page["section_id"],
            page_content=page["page_content"],
            rerank_score=float(score),
            outbound_citations=page.get("outbound_citations", []),
            sha256=page.get("sha256", ""),
            md5=page.get("md5", ""),
            metadata={k: v for k, v in page.items() if not k.startswith("_")},
        )

    def retrieve(
        self, query: str, audit_date: str | None = None, mode: str = BACKEND_NAME, top_k: int = DEFAULT_RETRIEVAL_TOP_K, **kwargs: Any
    ) -> RetrievalResponse:
        """Retrieve `top_k` relevant pages for `query` and optional ISO `audit_date`.

        Args:
            query: Compliance question string.
            audit_date: Optional `YYYY-MM-DD` evaluation date.
            mode: Retrieval/ablation mode identifier.
            top_k: Maximum parent page chunks to return.
        Returns:
            `RetrievalResponse` containing ranked `RetrievedParentChunk` items.
        """
        validated_date = _validate_audit_date(audit_date) if (audit_date and audit_date.strip()) else None
        if mode in UNSUPPORTED_DENSE_MODES:
            return RetrievalResponse(query=query, audit_date=audit_date, mode=mode, parent_chunks=[], sub_queries=[query])

        active_date = validated_date if mode not in NON_TEMPORAL_ABLATION_MODES else None
        candidate_pages: list[dict[str, Any]] = []
        for doc_node in self.tree.values():
            if not active_date or (doc_node["valid_from"] <= active_date <= doc_node["valid_to"]):
                candidate_pages.extend(doc_node.get("pages", []))
        if not candidate_pages:
            return RetrievalResponse(query=query, audit_date=audit_date, mode=mode, parent_chunks=[], sub_queries=[query])

        sub_queries = decompose_compound_query(query)
        target_doc_id = infer_target_doc_id(query)
        main_tokens, main_targets = _tokenize(query), parse_query_targets(query, target_doc_id)
        global_scores: dict[str, float] = {}
        sub_scores: list[dict[str, float]] = [{} for _ in sub_queries]
        sub_tokens_list = [_tokenize(sq) for sq in sub_queries]
        sub_target_docs = [infer_target_doc_id(sq) or target_doc_id for sq in sub_queries]
        sub_targets_list = [parse_query_targets(sq, sub_target_docs[i]) for i, sq in enumerate(sub_queries)]

        for page in candidate_pages:
            pid = page["parent_id"]
            c_low, sec_low, sum_low = self._content_low.get(pid), self._section_low.get(pid), self._summary_low.get(pid)
            best_g = self._bm25_score(main_tokens, pid) + compute_tree_navigation_score(
                query, page, target_doc_id, targets=main_targets, content_low=c_low, section_low=sec_low, summary_low=sum_low
            )
            for sq_i, sq in enumerate(sub_queries):
                sq_total = self._bm25_score(sub_tokens_list[sq_i], pid) + compute_tree_navigation_score(
                    sq, page, sub_target_docs[sq_i], targets=sub_targets_list[sq_i], content_low=c_low, section_low=sec_low, summary_low=sum_low
                )
                sub_scores[sq_i][pid] = sq_total
                if sq_total > best_g:
                    best_g = sq_total
            global_scores[pid] = best_g

        selected_pids: list[tuple[str, float]] = []
        seen_pids: set[str] = set()
        if len(sub_queries) > 1:
            for sq_i in range(len(sub_queries)):
                for pid, _ in sorted(sub_scores[sq_i].items(), key=lambda x: (-x[1], x[0])):
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        selected_pids.append((pid, global_scores[pid]))
                        break
        for pid, sc in sorted(global_scores.items(), key=lambda x: (-x[1], x[0])):
            if len(selected_pids) >= top_k:
                break
            if pid not in seen_pids:
                seen_pids.add(pid)
                selected_pids.append((pid, sc))

        parent_chunks = [self._to_parent_chunk(self.page_by_id[pid], sc) for pid, sc in selected_pids[:top_k]]
        return RetrievalResponse(query=query, audit_date=audit_date, mode=mode, parent_chunks=parent_chunks, sub_queries=sub_queries)
