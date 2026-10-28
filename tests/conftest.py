"""
Shared pytest fixtures for unit and software conformance tests in tests/.

Provides session-scoped fixtures:
- `ingestor`: Instance of ComplianceCorpusIngestor for corpus verification.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Strictly enforce isolated test environment (.env.test) for all unit and technical tests
os.environ["ENV_FILE"] = ".env.test"

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = WORKSPACE_ROOT / "frontend"
for p in (WORKSPACE_ROOT, FRONTEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rag_engine.backends.vector_chunking.ingest import ComplianceCorpusIngestor


@pytest.fixture(scope="session")
def ingestor() -> ComplianceCorpusIngestor:
    """Session-scoped fixture providing an indexed ComplianceCorpusIngestor for testing."""
    ing = ComplianceCorpusIngestor()
    ing.ingest(clean=False)
    return ing
