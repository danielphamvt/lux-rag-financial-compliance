"""Shared runtime bootstrap and JSON loading for evaluation modules.

Centralizes the setup every evaluation entry point repeats: pinning ENV_FILE
to the isolated test environment and making the workspace root importable
(pytest.ini's `pythonpath` covers this under pytest; CLI runners need it too).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def ensure_runtime() -> None:
    """Pin ENV_FILE to the test env (unless already set) and add the workspace root to sys.path.

    An explicit ENV_FILE from the environment (Makefile targets, `--env-file`)
    always wins over the .env.test default.
    """
    os.environ.setdefault("ENV_FILE", ".env.test")
    root = str(WORKSPACE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def read_json(path: Path) -> Any:
    """Load a JSON file, raising on missing or malformed input — no silent fallback.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON content (object, array, or scalar).
    """
    return json.loads(path.read_text(encoding="utf-8"))
