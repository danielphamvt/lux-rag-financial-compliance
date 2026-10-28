"""Backend registry: name -> dotted module path.

Lazy import inside get_backend() so that list_backends() (and config-time code)
never loads torch/qdrant/model weights.
"""
from __future__ import annotations

import importlib
import os

from rag_engine.core.interface import ComplianceBackend

DEFAULT_BACKEND = "vector_chunking"

# Registering a new backend = add one line here (see CLAUDE.md "Adding a backend").
BACKENDS: dict[str, str] = {
    "vector_chunking": "rag_engine.backends.vector_chunking",
    "page_index": "rag_engine.backends.page_index",
}


def list_backends() -> list[str]:
    return sorted(BACKENDS)


def get_backend(name: str | None = None) -> ComplianceBackend:
    selected = name or os.getenv("RAG_BACKEND", DEFAULT_BACKEND)
    if selected not in BACKENDS:
        raise ValueError(
            f"Unknown backend {selected!r}. Registered backends: {', '.join(sorted(BACKENDS))}"
        )
    module = importlib.import_module(BACKENDS[selected])
    backend = module.create_backend()
    if not callable(getattr(backend, "retrieve", None)):
        raise TypeError(f"Backend {selected!r} does not expose retrieve().")
    return backend
