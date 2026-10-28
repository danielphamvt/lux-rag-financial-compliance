"""Centralized, scalable configuration for the `vector_chunking` RAG backend.

Environment variable policy:
- `.env` is reserved for global infrastructure (LLM provider/API keys, `RAG_BACKEND`, `DOCS_DIR`).
- Mode-specific tuning lives here and can optionally be overridden via `VECTOR_CHUNKING_*` env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag_engine.core.corpus_parsing import (
    CORPUS_MANIFEST_OVERRIDE_FILENAME,
    CORPUS_REGISTRY as _DEFAULT_CORPUS_REGISTRY,
    CSSF_02_77_SECTION_TITLES,
    CSSF_24_856_SECTION_TITLES,
    DEFAULT_JURISDICTION,
    DEFAULT_REGULATORY_AUTHORITY,
    DEFAULT_VALID_FROM,
    DEFAULT_VALID_TO,
    HASH_READ_BLOCK_SIZE,
    OUTBOUND_CITATION_PATTERNS,
    SUPPORTED_CORPUS_EXTENSIONS,
    UCITS_COLUMN_SPLIT_OFFSET,
    UCITS_HEADER_MIN_Y0,
    UCITS_MAX_ARTICLES_IN_SECTION_ID,
    UCITS_PAGE_23_SECTION_ID,
    CorpusDocumentSpec,
    discover_corpus_registry as _discover_corpus_registry,
)
from rag_engine.core.nav_audit_schemas import TWO_DECIMALS
from rag_engine.core.retrieval_types import (
    AUDIT_CONFIDENCE_SCORE,
    AUDIT_EXCERPT_MAX_CHARS,
    AUDIT_RETRIEVAL_TOP_K,
    BPS_MULTIPLIER,
    COMPOUND_QUERY_SEPARATORS,
    DOC_ROUTING_KEYWORDS,
    LOGS_DIR,
    NOMIC_SPAN_MAX_CHARS,
    NOMIC_SPAN_MIN_CHARS,
    WORKSPACE_ROOT,
    DomainConceptRule,
    _env_float,
    _env_int,
)

# --- 1. Retrieval Volume, Candidate Pool, Chunking & Structural Weights ---
DEFAULT_TOP_K: int = _env_int("VECTOR_CHUNKING_TOP_K", 5)
DEFAULT_RETRIEVAL_TOP_K: int = _env_int("VECTOR_CHUNKING_INTERNAL_TOP_K", 3)
DEFAULT_RETRIEVAL_MODE: str = os.getenv("VECTOR_CHUNKING_MODE", "hybrid_rerank")
MIN_RELEVANCE_SCORE: float = _env_float("VECTOR_CHUNKING_MIN_SCORE", 0.75)
DEFAULT_CANDIDATE_K: int = _env_int("VECTOR_CHUNKING_CANDIDATE_K", 12)
SCOPED_DOC_CANDIDATE_LIMIT: int = _env_int("VECTOR_CHUNKING_SCOPED_CANDIDATE_K", 8)
MAX_RERANK_CANDIDATES: int = _env_int("VECTOR_CHUNKING_MAX_RERANK_CANDIDATES", 8)
MAX_QUERY_CACHE_SIZE: int = _env_int("VECTOR_CHUNKING_MAX_QUERY_CACHE_SIZE", 512)

CHILD_CHUNK_SIZE: int = _env_int("VECTOR_CHUNKING_CHILD_CHUNK_SIZE", 550)
CHILD_CHUNK_OVERLAP: int = _env_int("VECTOR_CHUNKING_CHILD_CHUNK_OVERLAP", 100)
CHILD_CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
ATOMIC_TABLE_MAX_CHARS: int = _env_int("VECTOR_CHUNKING_ATOMIC_TABLE_MAX_CHARS", 2400)

WEIGHT_HEADER_TABLE_MATCH: float = _env_float("VECTOR_CHUNKING_WEIGHT_HEADER_TABLE", 1.5)
WEIGHT_BODY_TABLE_MATCH: float = _env_float("VECTOR_CHUNKING_WEIGHT_BODY_TABLE", 0.6)
WEIGHT_HEADER_CLAUSE_MATCH: float = _env_float("VECTOR_CHUNKING_WEIGHT_HEADER_CLAUSE", 1.5)
WEIGHT_BODY_CLAUSE_MATCH: float = _env_float("VECTOR_CHUNKING_WEIGHT_BODY_CLAUSE", 0.8)
WEIGHT_NUMBERED_POINT_MATCH: float = _env_float("VECTOR_CHUNKING_WEIGHT_POINT_MATCH", 1.4)
NON_ATOMIC_CHUNK_PRIOR_BONUS: float = _env_float("VECTOR_CHUNKING_NON_ATOMIC_PRIOR", 0.05)
SUB_QUERY_STAGE1_BONUS: float = _env_float("VECTOR_CHUNKING_SUB_QUERY_BONUS", 0.5)
DOMAIN_CONCEPT_BOOST_SCALE: float = _env_float("VECTOR_CHUNKING_DOMAIN_BOOST_SCALE", 1.0)

EMBEDDING_MODEL_NAME: str = os.getenv("VECTOR_CHUNKING_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
SPARSE_MODEL_NAME: str = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
EMBEDDING_ENCODE_BATCH_SIZE: int = _env_int("VECTOR_CHUNKING_EMBED_BATCH_SIZE", 8)
QDRANT_INDEX_BATCH_SIZE: int = _env_int("VECTOR_CHUNKING_QDRANT_BATCH_SIZE", 16)
RERANKER_BATCH_SIZE: int = _env_int("VECTOR_CHUNKING_RERANK_BATCH_SIZE", 8)
RERANKER_MAX_LENGTH: int = _env_int("VECTOR_CHUNKING_RERANK_MAX_LEN", 256)
RERANKER_QUERY_MAX_CHARS: int = _env_int("VECTOR_CHUNKING_RERANK_QUERY_CHARS", 550)
RERANKER_DOC_MAX_CHARS: int = _env_int("VECTOR_CHUNKING_RERANK_DOC_CHARS", 950)

# --- 2. Storage Paths, Qdrant Collections, Static Buckets & Runtime Settings ---
RUNTIME_ENV_DEFAULTS: dict[str, str] = {
    "MallocStackLogging": "0",
    "MallocStackLoggingNoCompact": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
BACKEND_NAME: str = "vector_chunking"
REQUIRES_WARMUP: bool = True
WARMUP_QUERY: str = "management fee tolerance threshold"
WARMUP_AUDIT_DATE: str = "2025-02-15"
WARMUP_TOP_K: int = 1

DATA_DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", str(WORKSPACE_ROOT / "data_docs")))
FALLBACK_DATA_DOCS_DIR: Path = WORKSPACE_ROOT / "data_docs"
STORAGE_DIR: Path = Path(os.getenv("VECTOR_CHUNKING_STORAGE_DIR", str(WORKSPACE_ROOT / "data" / "compliance_rag")))
MARKDOWN_CACHE_DIR: Path = STORAGE_DIR / "markdown_docs"
PARENT_STORE_DIR: Path = STORAGE_DIR / "parent_store"
QDRANT_DB_DIR: Path = STORAGE_DIR / "qdrant_db"
COLLECTION_NAME: str = "nav_compliance_children"
SPARSE_VECTOR_NAME: str = "sparse"
MANIFEST_FILENAME: str = "indexing_manifest.json"
QDRANT_SNAPSHOT_PREFIX: str = "qdrant_snapshot_"

AZURE_API_VERSION_DEFAULT: str = "2024-02-01"
DIMENSION_PROBE_QUERY: str = "NAV compliance audit test"
DENSE_TOKEN_BUCKETS: tuple[int, ...] = (48, 80, 112, 160, 256)
DENSE_DEFAULT_BUCKET_LEN: int = 256
RERANKER_TOKEN_BUCKETS: tuple[int, ...] = (192, 256)
RERANKER_SHORT_BUCKET_CHAR_THRESHOLD: int = 920

# --- Nomic v1.5 Baseline Ablation Settings ---
NOMIC_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_COLLECTION_NAME: str = "nav_compliance_children_nomic"
NOMIC_QDRANT_DB_DIR: Path = STORAGE_DIR / "qdrant_nomic_db"
NOMIC_SNAPSHOT_PREFIX: str = "qdrant_nomic_snapshot_"
NOMIC_VECTOR_DIM: int = 768
NOMIC_THREADS: int = 2
NOMIC_EMBED_BATCH_SIZE: int = 16
NOMIC_SCROLL_LIMIT: int = 2500
NOMIC_CHILD_MAX_CHARS: int = 900
NOMIC_DOC_PREFIX: str = "search_document: "
NOMIC_QUERY_PREFIX: str = "search_query: "

# --- 3. Corpus Discovery & Bitemporal Catalog ---
CORPUS_REGISTRY: dict[str, CorpusDocumentSpec] = dict(_DEFAULT_CORPUS_REGISTRY)


def discover_corpus_registry(
    primary_dir: Path = DATA_DOCS_DIR,
    fallback_dir: Path = FALLBACK_DATA_DOCS_DIR,
) -> dict[str, CorpusDocumentSpec]:
    """Build the document registry from `CORPUS_REGISTRY`, `corpus_manifest.json`, and directory scan."""
    return _discover_corpus_registry(primary_dir, fallback_dir, default_registry=CORPUS_REGISTRY)


# --- 4. Declarative Domain Rules ---
DOMAIN_CONCEPT_RULES: tuple[DomainConceptRule, ...] = (
    DomainConceptRule(
        name="customized_tolerance_governance",
        weight=2.0,
        query_any=("customized tolerance", "customised tolerance", "alternative tolerance", "maximum tolerance"),
        content_groups=(("customized", "customised"), ("tolerance", "nav")),
        section_any=("tolerance",),
    ),
    DomainConceptRule(
        name="mmf_cash_uci_tolerance_taxonomy",
        weight=0.8,
        query_any=("money market", "mmf", "cash fund", "cash uci"),
        content_groups=(("money market", "mmf", "cash fund"), ("tolerance", "threshold", "% of nav", "bps")),
    ),
    DomainConceptRule(
        name="bond_mixed_uci_tolerance_taxonomy",
        weight=1.8,
        query_any=("bond uci", "bond ucits", "bond sif", "mixed uci", "mixed ucits", "fixed income", "multi-asset"),
        content_groups=(("bond ucis", "mixed ucis", "mixed investment policy"), ("tolerance", "threshold", "% of nav", "bps")),
    ),
    DomainConceptRule(
        name="equity_other_uci_tolerance_taxonomy",
        weight=1.5,
        query_any=("shares and other financial assets", "equity fund", "equity uci", "equity ucits"),
        content_groups=(("shares and other financial assets", "equity ucis", "equity funds"), ("tolerance", "threshold", "% of nav", "bps")),
    ),
    DomainConceptRule(
        name="corrective_action_plan_exemption",
        weight=1.2,
        query_any=("corrective action plan", "indemnification", "reimbursed"),
        content_groups=(("corrective action plan",), ("does not exceed",)),
    ),
    DomainConceptRule(
        name="de_minimis_threshold_governance",
        weight=1.2,
        query_any=("de minimis",),
        content_groups=(("de minimis",),),
    ),
    DomainConceptRule(
        name="successive_aggregate_error_assessment",
        weight=1.2,
        query_any=("simultaneous", "successive", "aggregate"),
        content_groups=(("simultaneous", "successive"), ("aggregate",)),
    ),
    DomainConceptRule(
        name="external_auditor_special_report_controls",
        weight=1.2,
        query_any=("compensation thresholds", "separate/special report", "special report", "separate report", "annual audit"),
        content_groups=(("annual audit", "50,000", "25,000"), ("special report", "separate report")),
        section_any=("rea", "summary tables"),
    ),
    DomainConceptRule(
        name="investment_breach_vs_nav_error_scope",
        weight=1.5,
        query_any=("non-compliance with investment rules", "active non-compliance", "passive non-compliance", "investment breach"),
        content_groups=(("non-compliance",), ("investment rules",), ("cannot be applied", "tolerance", "active")),
    ),
    DomainConceptRule(
        name="share_class_fee_and_tax_schedule",
        weight=1.5,
        query_any=("management fee", "fee schedule", "share class", "taxe d'abonnement", "subscription tax", "footnote", "fee rate", "fee reduction"),
        content_groups=(("per annum",), ("bps", "taxe d'abonnement", "subscription tax")),
    ),
    DomainConceptRule(
        name="accrual_day_count_convention",
        weight=1.5,
        query_any=("day-count", "day count", "accrual convention", "accrues", "accrual"),
        content_groups=(("day-count", "day count", "accrual"), ("accrual", "accrued", "calendar days")),
    ),
)


# --- 5. Structured Configuration Dataclass ---
@dataclass(frozen=True)
class VectorChunkingConfig:
    """Typed, immutable configuration container for the `vector_chunking` backend."""

    backend_name: str = BACKEND_NAME
    default_top_k: int = DEFAULT_TOP_K
    default_retrieval_mode: str = DEFAULT_RETRIEVAL_MODE
    min_relevance_score: float = MIN_RELEVANCE_SCORE
    weight_header_table_match: float = WEIGHT_HEADER_TABLE_MATCH
    weight_header_clause_match: float = WEIGHT_HEADER_CLAUSE_MATCH
    domain_concept_boost_scale: float = DOMAIN_CONCEPT_BOOST_SCALE
    requires_warmup: bool = REQUIRES_WARMUP
    warmup_query: str = WARMUP_QUERY
    warmup_audit_date: str = WARMUP_AUDIT_DATE
    warmup_top_k: int = WARMUP_TOP_K
    data_docs_dir: Path = DATA_DOCS_DIR
    storage_dir: Path = STORAGE_DIR
    collection_name: str = COLLECTION_NAME

    @property
    def min_score(self) -> float:
        """Alias for `min_relevance_score` matching the `ComplianceBackend` interface attribute."""
        return self.min_relevance_score


VECTOR_CHUNKING_CONFIG: VectorChunkingConfig = VectorChunkingConfig()
