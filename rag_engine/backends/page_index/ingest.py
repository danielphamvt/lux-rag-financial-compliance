"""
SOTA Page-Level & Document-Tree Ingestion Pipeline (`page_index` backend).

Builds a deterministic hierarchical Document Tree (`Document -> Page`)
stored exclusively at `data/indexes/page_index/document_tree.json`, preserving
full pages, Markdown tables, and footnotes (`*`, `(*1)`, `(*2)`) without 512-token
fragmentation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .corpus_parsers import (
    _parse_cssf_02_77,
    _parse_cssf_24_856,
    _parse_generic_pdf,
    _parse_prospectus_markdown,
    _parse_ucits_directive,
    build_page_node,
    compute_file_hashes,
    extract_outbound_citations,
)
from .page_index_config import (
    BACKEND_NAME,
    DATA_DOCS_DIR,
    DOCUMENT_TREE_FILENAME,
    FALLBACK_DATA_DOCS_DIR,
    INDEX_DIR,
    TREE_SCHEMA_VERSION,
    CorpusDocumentSpec,
    discover_corpus_registry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PageIndexIngestor",
    "build_page_node",
    "compute_file_hashes",
    "extract_outbound_citations",
]


class PageIndexIngestor:
    """Ingests regulatory PDFs and Prospectus Markdown into a hierarchical Document Tree JSON."""

    def __init__(
        self,
        data_docs_dir: Path = DATA_DOCS_DIR,
        index_dir: Path = INDEX_DIR,
    ):
        self.data_docs_dir = Path(data_docs_dir)
        self.index_dir = Path(index_dir)
        self.tree_path = self.index_dir / DOCUMENT_TREE_FILENAME

    def _resolve_corpus_file(self, filename: str) -> Path:
        file_path = self.data_docs_dir / filename
        if file_path.exists():
            return file_path
        fallback_path = FALLBACK_DATA_DOCS_DIR / filename
        if fallback_path.exists():
            return fallback_path
        raise FileNotFoundError(f"Required corpus file missing: {file_path}")

    def _is_cache_valid(
        self,
        tree_data: dict[str, Any],
        active_registry: dict[str, CorpusDocumentSpec],
    ) -> bool:
        """Verify cached document_tree.json matches current schema version, corpus set, and file hashes."""
        if tree_data.get("version") != TREE_SCHEMA_VERSION:
            return False
        manifest = tree_data.get("manifest")
        documents = tree_data.get("documents")
        if not isinstance(manifest, dict) or not isinstance(documents, dict) or not documents:
            return False

        manifest_docs = manifest.get("documents")
        if not isinstance(manifest_docs, dict):
            return False

        expected_doc_ids = {spec.doc_id for spec in active_registry.values()}
        if set(manifest_docs.keys()) != expected_doc_ids or set(documents.keys()) != expected_doc_ids:
            return False

        for filename, spec in active_registry.items():
            try:
                file_path = self._resolve_corpus_file(filename)
            except FileNotFoundError:
                return False
            hashes = compute_file_hashes(file_path)
            cached_doc = manifest_docs.get(spec.doc_id, {})
            doc_tree_node = documents.get(spec.doc_id, {})
            if (
                cached_doc.get("filename") != spec.filename
                or cached_doc.get("valid_from") != spec.valid_from
                or cached_doc.get("valid_to") != spec.valid_to
                or cached_doc.get("sha256") != hashes["sha256"]
                or "sections" in doc_tree_node
            ):
                return False
        return True

    def ingest(self, clean: bool = False) -> dict[str, Any]:
        """Ingest all discovered corpus documents into document_tree.json.

        Args:
            clean: When True, force a full rebuild ignoring any existing cache.

        Returns:
            Ingestion manifest dictionary summarizing indexed documents and page counts.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        active_registry = discover_corpus_registry(self.data_docs_dir, FALLBACK_DATA_DOCS_DIR)

        if not clean and self.tree_path.exists():
            try:
                tree_data = json.loads(self.tree_path.read_text(encoding="utf-8"))
                if self._is_cache_valid(tree_data, active_registry):
                    return tree_data["manifest"]
            except (json.JSONDecodeError, OSError):
                logger.warning("Cached document_tree.json is unreadable; rebuilding index.")

        documents_tree: dict[str, Any] = {}
        manifest_docs: dict[str, Any] = {}
        total_pages = 0

        for filename, spec in active_registry.items():
            file_path = self._resolve_corpus_file(filename)
            hashes = compute_file_hashes(file_path)
            if filename == "cssf02_77eng.pdf":
                pages = _parse_cssf_02_77(file_path, spec, hashes)
            elif filename == "CSSF24_856eng.pdf":
                pages = _parse_cssf_24_856(file_path, spec, hashes)
            elif filename == "UCITS_Directive_2009_65_EC.pdf":
                pages = _parse_ucits_directive(file_path, spec, hashes)
            elif filename.lower().endswith(".md"):
                pages = _parse_prospectus_markdown(file_path, spec, hashes)
            elif filename.lower().endswith(".pdf"):
                pages = _parse_generic_pdf(file_path, spec, hashes)
            else:
                raise ValueError(f"Unhandled corpus file: {filename}")

            documents_tree[spec.doc_id] = {
                "doc_id": spec.doc_id,
                "filename": spec.filename,
                "title": spec.title,
                "valid_from": spec.valid_from,
                "valid_to": spec.valid_to,
                "sha256": hashes["sha256"],
                "md5": hashes["md5"],
                "page_count": len(pages),
                "pages": pages,
            }
            manifest_docs[spec.doc_id] = {
                "filename": spec.filename,
                "doc_id": spec.doc_id,
                "valid_from": spec.valid_from,
                "valid_to": spec.valid_to,
                "sha256": hashes["sha256"],
                "md5": hashes["md5"],
                "page_count": len(pages),
                "parent_chunk_count": len(pages),
                "child_chunk_count": len(pages),
            }
            total_pages += len(pages)

        manifest: dict[str, Any] = {
            "backend": BACKEND_NAME,
            "total_documents": len(documents_tree),
            "total_pages": total_pages,
            "total_parent_chunks": total_pages,
            "total_child_chunks": total_pages,
            "index_path": str(self.tree_path),
            "documents": manifest_docs,
        }

        payload = {
            "version": TREE_SCHEMA_VERSION,
            "manifest": manifest,
            "documents": documents_tree,
        }
        self.tree_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into SOTA Page Index Document Tree.")
    parser.add_argument("--clean", action="store_true", help="Force clean re-ingestion.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ingestor = PageIndexIngestor()
    result_manifest = ingestor.ingest(clean=args.clean)
    print(json.dumps(result_manifest, indent=2))
