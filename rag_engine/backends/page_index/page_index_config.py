"""Centralized configuration for the `page_index` RAG backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag_engine.core.corpus_parsing import (
    CORPUS_MANIFEST_OVERRIDE_FILENAME,
    CORPUS_REGISTRY,
    CorpusDocumentSpec,
    DEFAULT_VALID_FROM,
    DEFAULT_VALID_TO,
    HASH_READ_BLOCK_SIZE,
    SUPPORTED_CORPUS_EXTENSIONS,
    discover_corpus_registry as _discover_corpus_registry,
)
from rag_engine.core.retrieval_types import (
    DOC_ROUTING_KEYWORDS,
    DomainConceptRule,
    WORKSPACE_ROOT,
    _env_float,
    _env_int,
)

# --- 1. Retrieval, BM25, Structural Tree-Scoring & Layout Hyperparameters ---
DEFAULT_TOP_K: int = _env_int("PAGE_INDEX_TOP_K", 5)
DEFAULT_RETRIEVAL_TOP_K: int = _env_int("PAGE_INDEX_INTERNAL_TOP_K", 3)
MIN_RELEVANCE_SCORE: float = _env_float("PAGE_INDEX_MIN_SCORE", 15.0)

BM25_K1: float = _env_float("PAGE_INDEX_BM25_K1", 1.5)
BM25_B: float = _env_float("PAGE_INDEX_BM25_B", 0.75)
MIN_TOKEN_LENGTH: int = _env_int("PAGE_INDEX_MIN_TOKEN_LENGTH", 2)
TOKEN_REGEX: str = r"[a-z0-9]+(?:\.[0-9]+)?%?"

WEIGHT_TARGET_DOC_MATCH: float = _env_float("PAGE_INDEX_WEIGHT_DOC_MATCH", 6.0)
WEIGHT_HEADER_TABLE_MATCH: float = _env_float("PAGE_INDEX_WEIGHT_HEADER_TABLE", 8.0)
WEIGHT_BODY_TABLE_MATCH: float = _env_float("PAGE_INDEX_WEIGHT_BODY_TABLE", 3.0)
WEIGHT_HEADER_CLAUSE_MATCH: float = _env_float("PAGE_INDEX_WEIGHT_HEADER_CLAUSE", 8.0)
WEIGHT_BODY_CLAUSE_MATCH: float = _env_float("PAGE_INDEX_WEIGHT_BODY_CLAUSE", 4.0)
WEIGHT_NUMBERED_POINT_MATCH: float = _env_float("PAGE_INDEX_WEIGHT_POINT_MATCH", 7.0)
MAX_POINT_RANGE_SPAN: int = _env_int("PAGE_INDEX_MAX_POINT_SPAN", 12)
DOMAIN_CONCEPT_BOOST_SCALE: float = _env_float("PAGE_INDEX_DOMAIN_BOOST_SCALE", 1.0)

MAX_HEADING_CHARS: int = _env_int("PAGE_INDEX_MAX_HEADING_CHARS", 120)
MAX_POINTS_PER_PAGE: int = _env_int("PAGE_INDEX_MAX_POINTS_PER_PAGE", 15)
MAX_PERCENTAGES_PER_PAGE: int = _env_int("PAGE_INDEX_MAX_PCT_PER_PAGE", 8)
MAX_EUR_AMOUNTS_PER_PAGE: int = _env_int("PAGE_INDEX_MAX_EUR_PER_PAGE", 6)
MAX_SUMMARY_HEADINGS: int = _env_int("PAGE_INDEX_MAX_SUMMARY_HEADINGS", 5)
MAX_SUMMARY_KEYWORDS: int = _env_int("PAGE_INDEX_MAX_SUMMARY_KEYWORDS", 12)
UCITS_HEADER_MIN_Y0: float = _env_float("PAGE_INDEX_TWO_COL_MIN_Y0", 62.0)
UCITS_COLUMN_SPLIT_OFFSET: float = _env_float("PAGE_INDEX_TWO_COL_SPLIT_OFFSET", 10.0)
UCITS_MAX_ARTICLES_IN_SECTION_ID: int = _env_int("PAGE_INDEX_MAX_ARTICLES_IN_SEC_ID", 4)

# --- 2. Storage Paths, Ablation Modes & Backend Identity ---
BACKEND_NAME: str = "page_index"
REQUIRES_WARMUP: bool = False
DATA_DOCS_DIR: Path = Path(os.getenv("DOCS_DIR", str(WORKSPACE_ROOT / "data_docs")))
FALLBACK_DATA_DOCS_DIR: Path = WORKSPACE_ROOT / "data_docs"
INDEX_DIR: Path = Path(os.getenv("PAGE_INDEX_DIR", str(WORKSPACE_ROOT / "data" / "indexes" / "page_index")))
DOCUMENT_TREE_FILENAME: str = "document_tree.json"
DOCUMENT_TREE_PATH: Path = INDEX_DIR / DOCUMENT_TREE_FILENAME
TREE_SCHEMA_VERSION: str = "2.0"

UNSUPPORTED_DENSE_MODES: frozenset[str] = frozenset({"nomic_dense_only"})
NON_TEMPORAL_ABLATION_MODES: frozenset[str] = frozenset({"dense_only", "hybrid_no_temporal"})


def discover_corpus_registry(
    primary_dir: Path = DATA_DOCS_DIR,
    fallback_dir: Path = FALLBACK_DATA_DOCS_DIR,
) -> dict[str, CorpusDocumentSpec]:
    """Build document registry from `CORPUS_REGISTRY`, `corpus_manifest.json`, and directory scan."""
    return _discover_corpus_registry(primary_dir, fallback_dir, default_registry=CORPUS_REGISTRY)


CSSF_02_77_SECTION_TITLES: dict[int, str] = {
    2: "Introduction & Repeal of Circular CSSF 2000/8 (Page 2)",
    3: "Section I.2 — Materiality & NAV Error Tolerance Thresholds (Page 3)",
    4: "Section I.3(a)-(b) — Correction Procedures & EUR 25,000 / EUR 2,500 Indemnification Thresholds (Page 4)",
    5: "Section I.3(a)-(b) — Correction Procedures & EUR 25,000 / EUR 2,500 Indemnification Thresholds (Page 5)",
    7: "Section I.3(c)-(d) — De Minimis & External Auditor Review EUR 25,000 / EUR 2,500 (Page 7)",
    9: "Section I.3(c)-(d) — De Minimis & External Auditor Review EUR 25,000 / EUR 2,500 (Page 9)",
    10: "Section II — Non-Compliance with Investment Rules (Page 10)",
    11: "Section II — Non-Compliance with Investment Rules (Page 11)",
}

CSSF_24_856_SECTION_TITLES: dict[int, str] = {
    8: "Chapter 2 — Scope of Application (Points 1–5: UCITS, UCI Part II, SIFs, SICARs, MMFs, ELTIFs)",
    9: "Chapter 2 — Scope & Chapter 8 Exemptions (Points 6–7: EuVECA, EuSEF)",
    14: "Chapter 3 Section 3.5 — Role of the Depositary (Point 25: Notification to CSSF)",
    15: "Section 4.1 — Definition of an NAV calculation error (Paragraphs 28–30)",
    16: "Section 4.2(a) — Tolerance threshold for Money Market Funds (MMFs) in Table 1 (Paragraphs 31–35)",
    17: "Section 4.2(b)-(c) — Tolerance thresholds for UCITS (Table 2) and UCIs Part II / ELTIFs (Table 3) with Footnotes (*)",
    18: "Section 4.2 — Point 35(d)(i)-(iii) Customized Tolerance Thresholds & 5% of the NAV Maximum Cap (SIFs/SICARs/UCI Part II)",
    19: "Section 4.2 — Point 37(v) Foreign Country More Restrictive Tolerance Threshold",
    21: "Section 4.4 — Table 4 (Undervalued NAV) and Table 5 (Overvalued NAV) Investor & UCI Compensation Allocation",
    26: "Chapter 5 Section 5.4 — Correction of active non-compliance with investment rules (Paragraphs 65–71, Section 5.5 reference)",
    27: "Chapter 5 Section 5.5 — Correction methods for instances of non-compliance & impact calculation (Paragraphs 72–76)",
    28: "Chapter 5 Section 5.5.3 — Accounting Method (LIFO, FIFO, Weighted Average Cost) (Points 78–82)",
    29: "Chapter 5 Section 5.5.3 — Economic Method vs Accounting Method (Points 83–85)",
    31: "Chapter 6 Section 6.1 — Swing Pricing Errors Introduction (Paragraphs 96–99)",
    32: "Chapter 6 Section 6.1 — Incorrect application of Swing Pricing (Table 6 & Table 7 with Footnotes (*))",
    33: "Chapter 6 Section 6.1–6.2 — Swing Pricing (Table 8 with Footnote (*)) & Non-compliant payment of costs/fees",
    34: "Chapter 6 Section 6.2 — Non-compliant payment of costs/fees (Points 104–105)",
    35: "Chapter 6 Section 6.3–6.4 — Cut-off rules (Circular CSSF 04/146 Late Trading) & Share Class Hedging Allocation (Points 108–113)",
    36: "Chapter 6 Section 6.4 — Erroneously Allocated Investment Profit Retention (Points 114–115)",
    38: "Chapter 7 Section 7.2–7.4 — De Minimis Rule (Point 127) & Unclaimed Compensation Caisse de Consignation (Points 132–135)",
    39: "Chapter 8 Section 8.1 — REA Special Report Triggers EUR 50,000 / EUR 5,000 (Point 138)",
    42: "Chapter 8 Section 8.4 — Summary tables on REA intervention: Table 9 (EUR 50,000 / EUR 5,000 thresholds for separate/special report)",
    43: "Chapter 8 Section 8.4 — Summary tables on REA intervention: Table 10 & Table 11 (EUR 50,000 / EUR 5,000 thresholds)",
    44: "Chapter 9 — Notification to CSSF within 4 to 8 weeks (Point 157)",
    45: "Chapter 9 — Submission of REA Special Report within 3 months (Point 158)",
    46: "Chapter 10 — Transitional & Entry into Force Provisions (Paragraphs 162–165: 29 March 2024 to 1 January 2025)",
    47: "Chapter 10 — Repeal of Circular CSSF 02/77 (Paragraph 166)",
}

UCITS_PAGE_23_SECTION_ID: str = (
    "Chapter IV — Obligations Regarding the Depositary: Article 22 (22(1), 22(2), 22(3)(a)-(e)) & Articles 23–26 (Page 23)"
)

# --- 4. Stopwords, Query Decomposition & Declarative Domain Rules ---
STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "under", "what", "when", "where", "which", "who", "with",
})

COMPOUND_QUERY_SEPARATORS: tuple[str, ...] = (
    r",\s+and\s+(?:(?:for|to|in|under|by|from|on|at)\s+(?:[a-z0-9/-]+\s+){0,3},?\s*)?(?:what|which|who|whom|whose|where|when|how|does|do)\s+",
    r"\s+and\s+(?=(?:the|what|which)\s+[^?]*\b(?:table|section|article|chapter|point|paragraph)\s+\d+\b)",
)

OUTBOUND_CITATION_PATTERNS: tuple[str, ...] = (
    r"\bSection\s+\d+(?:\.\d+)*\b",
    r"\bSection\s+[IVX]+\.\s*\d+\.(?:\s*\([a-z]\))?",
    r"\bChapter\s+\d+(?:\.\d+)*\b",
    r"\bArticle\s+\d+(?:\(\d+\))?(?:\([a-z]\))?\b",
    r"\bTable\s+\d+\b",
    r"\bpoint\s+\d{1,3}\([a-z]\)(?:\([ivx]+\))?\b",
    r"\bCircular\s+CSSF\s+\d{2,4}/\d+\b",
    r"Footnote\s+`?\(\*\d*\)`?",
    r"\(\*1\)", r"\(\*2\)", r"\(\*\)",
)

DOMAIN_CONCEPT_RULES: tuple[DomainConceptRule, ...] = (
    DomainConceptRule(
        name="customized_tolerance_governance",
        weight=10.0,
        query_any=("customized tolerance", "customised tolerance", "alternative tolerance", "maximum tolerance"),
        content_groups=(("customized", "customised", "tolerance"), ("tolerance", "nav")),
        section_any=("tolerance",),
    ),
    DomainConceptRule(
        name="mmf_cash_uci_tolerance_taxonomy",
        weight=14.0,
        query_any=("money market", "mmf", "cash fund", "cash uci"),
        content_groups=(("money market", "mmf", "cash fund"), ("tolerance", "threshold", "% of nav", "bps")),
    ),
    DomainConceptRule(
        name="bond_mixed_uci_tolerance_taxonomy",
        weight=14.0,
        query_any=("bond uci", "bond ucits", "bond sif", "mixed uci", "mixed ucits", "fixed income", "multi-asset"),
        content_groups=(("bond", "mixed"), ("tolerance", "threshold", "% of nav", "bps")),
    ),
    DomainConceptRule(
        name="equity_other_uci_tolerance_taxonomy",
        weight=7.5,
        query_any=("shares and other financial assets", "equity fund", "equity uci", "equity ucits"),
        content_groups=(("shares and other financial assets", "equity"), ("tolerance", "threshold", "% of nav", "bps")),
    ),
    DomainConceptRule(
        name="corrective_action_plan_exemption",
        weight=14.0,
        query_any=("corrective action plan", "indemnification threshold", "reimbursed"),
        content_groups=(("corrective action plan",), ("does not exceed",)),
    ),
    DomainConceptRule(
        name="de_minimis_threshold_governance",
        weight=8.0,
        query_any=("de minimis",),
        content_groups=(("de minimis",),),
    ),
    DomainConceptRule(
        name="successive_aggregate_error_assessment",
        weight=9.0,
        query_any=("simultaneous", "successive", "aggregate"),
        content_groups=(("simultaneous", "successive"), ("aggregate",)),
    ),
    DomainConceptRule(
        name="special_report_compensation_controls",
        weight=9.0,
        query_any=("compensation thresholds", "separate/special report", "special report", "separate report", "annual audit"),
        content_groups=(("special report", "separate report"), ("annual audit", "50,000", "25,000")),
        section_any=("rea", "summary tables"),
    ),
    DomainConceptRule(
        name="investment_breach_vs_nav_error_scope",
        weight=10.0,
        query_any=("non-compliance with investment rules", "active non-compliance", "passive non-compliance", "investment breach"),
        content_groups=(("non-compliance",), ("investment rules",), ("cannot be applied", "tolerance")),
    ),
    DomainConceptRule(
        name="share_class_fee_and_tax_schedule",
        weight=9.0,
        query_any=("management fee", "fee schedule", "share class", "taxe d'abonnement", "subscription tax", "footnote"),
        content_groups=(("management fee", "taxe d'abonnement", "subscription tax"), ("share class", "per annum", "bps", "gross asset value")),
        section_any=("fee schedule",),
    ),
    DomainConceptRule(
        name="accrual_day_count_convention",
        weight=9.0,
        query_any=("day-count", "day count", "accrual convention"),
        content_groups=(("day-count", "day count", "calendar days"), ("accrual", "accrued", "calendar days")),
    ),
)


@dataclass(frozen=True)
class PageIndexConfig:
    """Typed, immutable configuration container for the `page_index` backend."""

    backend_name: str = BACKEND_NAME
    default_top_k: int = DEFAULT_TOP_K
    min_relevance_score: float = MIN_RELEVANCE_SCORE
    requires_warmup: bool = REQUIRES_WARMUP
    weight_target_doc_match: float = WEIGHT_TARGET_DOC_MATCH
    weight_header_clause_match: float = WEIGHT_HEADER_CLAUSE_MATCH
    domain_concept_boost_scale: float = DOMAIN_CONCEPT_BOOST_SCALE
    data_docs_dir: Path = DATA_DOCS_DIR
    index_dir: Path = INDEX_DIR

    @property
    def min_score(self) -> float:
        """Alias for `min_relevance_score` matching the `ComplianceBackend` interface attribute."""
        return self.min_relevance_score


PAGE_INDEX_CONFIG: PageIndexConfig = PageIndexConfig()
