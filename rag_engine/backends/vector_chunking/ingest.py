"""Structure-, Table-, Footnote-, and Bitemporal-Aware Ingestion Pipeline (`vector_chunking` backend).

Delegates raw PDF/Markdown extraction and corpus discovery to `rag_engine.core.corpus_parsing`
and manages Parent Store JSON persistence and Qdrant Hybrid (Dense + Sparse BM25) indexing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from rag_engine.core.corpus_parsing import (
    CorpusDocumentSpec, compute_file_hashes, extract_cssf_02_77_pages, extract_cssf_24_856_pages,
    extract_generic_pdf_pages, extract_outbound_citations, extract_prospectus_markdown_sections,
    extract_ucits_directive_pages,
)

from . import vector_chunking_config as _cfg
from .vector_chunking_config import (
    ATOMIC_TABLE_MAX_CHARS, AZURE_API_VERSION_DEFAULT, CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SEPARATORS,
    CHILD_CHUNK_SIZE, COLLECTION_NAME, CSSF_02_77_SECTION_TITLES, CSSF_24_856_SECTION_TITLES,
    DATA_DOCS_DIR, DIMENSION_PROBE_QUERY, EMBEDDING_ENCODE_BATCH_SIZE, EMBEDDING_MODEL_NAME,
    FALLBACK_DATA_DOCS_DIR, MANIFEST_FILENAME, MARKDOWN_CACHE_DIR, PARENT_STORE_DIR,
    QDRANT_INDEX_BATCH_SIZE, QDRANT_SNAPSHOT_PREFIX, RUNTIME_ENV_DEFAULTS,
    SPARSE_MODEL_NAME, SPARSE_VECTOR_NAME, STORAGE_DIR,
)

for _env_key, _env_val in RUNTIME_ENV_DEFAULTS.items():
    os.environ[_env_key] = _env_val

import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant.qdrant import RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


def get_optimal_device(model_type: str = "embedding") -> str:
    """Resolve optimal PyTorch device (`cuda`, `mps`, or `cpu`) with env override."""
    env_override = os.getenv(f"{model_type.upper()}_DEVICE") or os.getenv("DEVICE")
    if env_override and env_override.strip().lower() != "auto":
        return env_override.strip().lower()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_pdf_parents(
    pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str], extracted: list[tuple[int, str, str]],
    markdown_cache_dir: Path, *, include_page_in_header: bool, footnote_fn: Any, has_table_override: bool | None = None,
) -> list[Document]:
    """Build page-level Parent Chunks from extracted `(page_num, section_id, md)` tuples and cache Markdown."""
    parent_docs: list[Document] = []
    for page_num, section_id, md in extracted:
        metadata = {
            "parent_id": f"{spec.doc_id}_p{page_num:02d}", "doc_id": spec.doc_id,
            "source_file": spec.filename, "source": spec.filename, "title": spec.title,
            "valid_from": spec.valid_from, "valid_to": spec.valid_to,
            "valid_from_int": spec.valid_from_int, "valid_to_int": spec.valid_to_int,
            "page_number": page_num, "section_id": section_id,
            "outbound_citations": extract_outbound_citations(md),
            "has_table": ("|" in md and "---" in md) if has_table_override is None else has_table_override,
            "has_footnote": bool(footnote_fn(md)), "sha256": hashes["sha256"], "md5": hashes["md5"],
        }
        page_tag = f"Page {page_num}, " if include_page_in_header else ""
        header = f"### [{spec.doc_id}] {section_id} ({page_tag}Effective: {spec.valid_from} to {spec.valid_to})"
        parent_docs.append(Document(page_content=f"{header}\n\n{md}", metadata=metadata))
    markdown_cache_dir.mkdir(parents=True, exist_ok=True)
    full_md = "\n\n".join(f"<!-- PAGE {p} -->\n{md}" for p, _, md in extracted)
    (markdown_cache_dir / f"{pdf_path.stem}.md").write_text(full_md, encoding="utf-8")
    return parent_docs


def parse_cssf_02_77(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str], markdown_cache_dir: Path = MARKDOWN_CACHE_DIR) -> list[Document]:
    """Parse `cssf02_77eng.pdf` into page-level Parent Chunks."""
    return _build_pdf_parents(pdf_path, spec, hashes, extract_cssf_02_77_pages(pdf_path, CSSF_02_77_SECTION_TITLES), markdown_cache_dir, include_page_in_header=False, footnote_fn=lambda md: "(*)" in md or "1 In French" in md)


def parse_cssf_24_856(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str], markdown_cache_dir: Path = MARKDOWN_CACHE_DIR) -> list[Document]:
    """Parse `CSSF24_856eng.pdf` into page-level Parent Chunks with multi-page table stitching."""
    return _build_pdf_parents(pdf_path, spec, hashes, extract_cssf_24_856_pages(pdf_path, CSSF_24_856_SECTION_TITLES), markdown_cache_dir, include_page_in_header=True, footnote_fn=lambda md: "(*)" in md)


def parse_ucits_directive(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str], markdown_cache_dir: Path = MARKDOWN_CACHE_DIR) -> list[Document]:
    """Parse `UCITS_Directive_2009_65_EC.pdf` using Two-Column Official Journal layout reconstruction."""
    return _build_pdf_parents(pdf_path, spec, hashes, extract_ucits_directive_pages(pdf_path), markdown_cache_dir, include_page_in_header=False, footnote_fn=lambda _: False, has_table_override=False)


def parse_generic_pdf(pdf_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str], markdown_cache_dir: Path = MARKDOWN_CACHE_DIR) -> list[Document]:
    """Parse arbitrary non-baseline `.pdf` documents into page-level Parent Chunks."""
    return _build_pdf_parents(pdf_path, spec, hashes, extract_generic_pdf_pages(pdf_path), markdown_cache_dir, include_page_in_header=True, footnote_fn=lambda md: "(*)" in md or "(*1)" in md or "(*2)" in md)


def parse_prospectus_markdown(
    md_path: Path, spec: CorpusDocumentSpec, hashes: dict[str, str], markdown_cache_dir: Path = MARKDOWN_CACHE_DIR
) -> list[Document]:
    """Parse Prospectus or generic `.md` files into section-level Parent Chunks."""
    raw_text = md_path.read_text(encoding="utf-8")
    markdown_cache_dir.mkdir(parents=True, exist_ok=True)
    (markdown_cache_dir / md_path.name).write_text(raw_text, encoding="utf-8")
    doc_header, sections = extract_prospectus_markdown_sections(raw_text)
    parent_docs: list[Document] = []
    for idx, section_id, sub_text_clean in sections:
        header_line = f"### [{spec.doc_id}] {section_id} (Effective: {spec.valid_from} to {spec.valid_to})"
        full_content = f"{header_line}\n{doc_header}\n\n{sub_text_clean}" if doc_header else f"{header_line}\n\n{sub_text_clean}"
        metadata = {
            "parent_id": f"{spec.doc_id}_s{idx:02d}", "doc_id": spec.doc_id,
            "source_file": spec.filename, "source": spec.filename, "title": spec.title,
            "valid_from": spec.valid_from, "valid_to": spec.valid_to,
            "valid_from_int": spec.valid_from_int, "valid_to_int": spec.valid_to_int,
            "page_number": idx, "section_id": section_id,
            "share_classes": list(dict.fromkeys(re.findall(r"\bClass\s+[A-Z0-9-]+\b", sub_text_clean))),
            "outbound_citations": extract_outbound_citations(sub_text_clean),
            "has_table": "|" in sub_text_clean and "---" in sub_text_clean,
            "has_footnote": "(*1)" in sub_text_clean or "(*2)" in sub_text_clean,
            "sha256": hashes["sha256"], "md5": hashes["md5"],
        }
        parent_docs.append(Document(page_content=full_content, metadata=metadata))
    return parent_docs


def build_child_chunks(
    parent_docs: list[Document], child_chunk_size: int = CHILD_CHUNK_SIZE, child_chunk_overlap: int = CHILD_CHUNK_OVERLAP
) -> list[Document]:
    """Create Child Chunks from Parent Chunks with atomic table preservation and header prefixes."""
    splitter = RecursiveCharacterTextSplitter(
        separators=list(CHILD_CHUNK_SEPARATORS), chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )
    child_docs: list[Document] = []
    for p_doc in parent_docs:
        meta = dict(p_doc.metadata)
        prefix = f"[{meta['doc_id']} | {meta['section_id']} | Effective: {meta['valid_from']}..{meta['valid_to']}]\n"
        if meta.get("has_table") or meta.get("has_footnote"):
            atomic_meta = {**meta, "child_id": f"{meta['parent_id']}_atomic", "is_atomic_table_chunk": True}
            child_docs.append(Document(page_content=p_doc.page_content[:ATOMIC_TABLE_MAX_CHARS], metadata=atomic_meta))
        for c_idx, split_doc in enumerate(splitter.split_documents([p_doc])):
            c_meta = {**meta, "child_id": f"{meta['parent_id']}_c{c_idx:02d}", "is_atomic_table_chunk": False}
            content = split_doc.page_content
            if not content.startswith("### [") and not content.startswith("["):
                content = prefix + content
            child_docs.append(Document(page_content=content, metadata=c_meta))
    return child_docs


def open_qdrant_with_snapshot(db_dir: Path, prefix: str = QDRANT_SNAPSHOT_PREFIX) -> QdrantClient:
    """Open local Qdrant DB; fall back to an ephemeral snapshot copy if locked by another process."""
    try:
        return QdrantClient(path=str(db_dir))
    except RuntimeError as exc:
        if "already accessed by another instance" not in str(exc):
            raise
        import atexit
        import shutil
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        shutil.copytree(db_dir, tmp_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".lock"))
        atexit.register(lambda p=tmp_dir: shutil.rmtree(p, ignore_errors=True))
        logger.info("Qdrant DB %s locked; opened concurrent snapshot at %s", db_dir, tmp_dir)
        return QdrantClient(path=str(tmp_dir))


class ComplianceCorpusIngestor:
    """Manages end-to-end ingestion, Parent Store persistence, and Qdrant Hybrid Indexing."""

    def __init__(
        self, data_docs_dir: Path = DATA_DOCS_DIR, storage_dir: Path = STORAGE_DIR, collection_name: str = COLLECTION_NAME
    ):
        self.data_docs_dir, self.storage_dir, self.collection_name = Path(data_docs_dir), Path(storage_dir), collection_name
        self.parent_store_dir = self.storage_dir / "parent_store"
        self.qdrant_db_dir = self.storage_dir / "qdrant_db"
        self.markdown_cache_dir = self.storage_dir / "markdown_docs"
        self.parent_store_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_db_dir.mkdir(parents=True, exist_ok=True)

        dim_override = os.getenv("EMBEDDING_DIMENSIONS")
        dimensions = int(dim_override) if dim_override else None
        if os.getenv("AZURE_OPENAI_ENDPOINT") and (
            EMBEDDING_MODEL_NAME.startswith("text-embedding-3") or os.getenv("EMBEDDING_PROVIDER") == "azure"
        ):
            from langchain_openai import AzureOpenAIEmbeddings

            self.dense_embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", EMBEDDING_MODEL_NAME),
                api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION_DEFAULT),
                dimensions=dimensions,
            )
        elif EMBEDDING_MODEL_NAME.startswith("text-embedding-3") and os.getenv("OPENAI_API_KEY"):
            from langchain_openai import OpenAIEmbeddings

            self.dense_embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"), dimensions=dimensions
            )
        else:
            self._device = get_optimal_device("embedding")
            dense_dtype = torch.float16 if self._device in ("mps", "cuda") else torch.float32
            self.dense_embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": self._device, "model_kwargs": {"torch_dtype": dense_dtype}},
                encode_kwargs={"batch_size": EMBEDDING_ENCODE_BATCH_SIZE},
            )
        self.sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
        self.client = open_qdrant_with_snapshot(self.qdrant_db_dir)

    def _fallback_docs_dir(self) -> Path:
        return FALLBACK_DATA_DOCS_DIR if self.data_docs_dir == DATA_DOCS_DIR else self.data_docs_dir

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        if not self.client.collection_exists(self.collection_name):
            dim = len(self.dense_embeddings.embed_query(DIMENSION_PROBE_QUERY))
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                sparse_vectors_config={SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )

    def _resolve_corpus_file(self, filename: str) -> Path:
        for base in (self.data_docs_dir, self._fallback_docs_dir()):
            if (base / filename).exists():
                return base / filename
        raise FileNotFoundError(f"Required corpus file missing: {self.data_docs_dir / filename}")

    def _is_cache_valid(self, existing_manifest: dict[str, Any], active_registry: dict[str, CorpusDocumentSpec]) -> bool:
        """Verify cached manifest, parent JSON store, and Qdrant index match current corpus files."""
        manifest_docs = existing_manifest.get("documents") if isinstance(existing_manifest, dict) else None
        if not isinstance(manifest_docs, dict) or set(manifest_docs.keys()) != {s.doc_id for s in active_registry.values()}:
            return False
        if not self.client.collection_exists(self.collection_name):
            return False
        count = self.client.count(self.collection_name).count
        if count <= 0 or existing_manifest.get("total_child_chunks") != count:
            return False
        parent_files = [p for p in self.parent_store_dir.glob("*.json") if p.name != MANIFEST_FILENAME]
        if len(parent_files) != existing_manifest.get("total_parent_chunks"):
            return False
        for filename, spec in active_registry.items():
            try:
                hashes = compute_file_hashes(self._resolve_corpus_file(filename))
            except FileNotFoundError:
                return False
            cached = manifest_docs.get(spec.doc_id, {})
            if (cached.get("filename"), cached.get("valid_from"), cached.get("valid_to"), cached.get("sha256")) != (
                spec.filename, spec.valid_from, spec.valid_to, hashes["sha256"]
            ):
                return False
        return True

    def parse_all_documents(self) -> tuple[list[Document], list[Document], dict[str, Any]]:
        doc_parents: dict[str, list[Document]] = {}
        doc_info: dict[str, dict[str, Any]] = {}
        active_registry = _cfg.discover_corpus_registry(self.data_docs_dir, self._fallback_docs_dir())
        for filename, spec in active_registry.items():
            file_path = self._resolve_corpus_file(filename)
            hashes = compute_file_hashes(file_path)
            if filename == "cssf02_77eng.pdf":
                parents = parse_cssf_02_77(file_path, spec, hashes, self.markdown_cache_dir)
            elif filename == "CSSF24_856eng.pdf":
                parents = parse_cssf_24_856(file_path, spec, hashes, self.markdown_cache_dir)
            elif filename == "UCITS_Directive_2009_65_EC.pdf":
                parents = parse_ucits_directive(file_path, spec, hashes, self.markdown_cache_dir)
            elif filename.lower().endswith(".md"):
                parents = parse_prospectus_markdown(file_path, spec, hashes, self.markdown_cache_dir)
            elif filename.lower().endswith(".pdf"):
                parents = parse_generic_pdf(file_path, spec, hashes, self.markdown_cache_dir)
            else:
                raise ValueError(f"Unhandled corpus file: {filename}")
            doc_parents[spec.doc_id] = parents
            doc_info[spec.doc_id] = {
                "filename": filename, "doc_id": spec.doc_id, "valid_from": spec.valid_from, "valid_to": spec.valid_to,
                "sha256": hashes["sha256"], "md5": hashes["md5"], "parent_chunk_count": len(parents),
            }

        all_parents = [p for parents in doc_parents.values() for p in parents]
        all_children = build_child_chunks(all_parents)
        child_counts: dict[str, int] = {}
        for child in all_children:
            child_counts[child.metadata["doc_id"]] = child_counts.get(child.metadata["doc_id"], 0) + 1
        for doc_id, info in doc_info.items():
            info["child_chunk_count"] = child_counts.get(doc_id, 0)
        return all_parents, all_children, {
            "documents": doc_info, "total_parent_chunks": len(all_parents), "total_child_chunks": len(all_children)
        }

    def ingest(self, clean: bool = False) -> dict[str, Any]:
        import gc

        manifest_path = self.parent_store_dir / MANIFEST_FILENAME
        active_registry = _cfg.discover_corpus_registry(self.data_docs_dir, self._fallback_docs_dir())
        if not clean and manifest_path.exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if self._is_cache_valid(existing_manifest, active_registry):
                    return existing_manifest
            except (json.JSONDecodeError, OSError):
                logger.warning("Cached indexing_manifest.json is unreadable; rebuilding index.")

        parents, children, manifest = self.parse_all_documents()
        for old_file in self.parent_store_dir.glob("*.json"):
            old_file.unlink()
        for p_doc in parents:
            out_path = self.parent_store_dir / f"{p_doc.metadata['parent_id']}.json"
            out_path.write_text(
                json.dumps({"page_content": p_doc.page_content, "metadata": p_doc.metadata}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        self.ensure_collection(recreate=True)
        vector_store = QdrantVectorStore(
            client=self.client, collection_name=self.collection_name, embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings, retrieval_mode=RetrievalMode.HYBRID, sparse_vector_name=SPARSE_VECTOR_NAME,
        )
        for idx in range(0, len(children), QDRANT_INDEX_BATCH_SIZE):
            vector_store.add_documents(children[idx : idx + QDRANT_INDEX_BATCH_SIZE])
            gc.collect()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(json.dumps(ComplianceCorpusIngestor().ingest(clean=True), indent=2))
