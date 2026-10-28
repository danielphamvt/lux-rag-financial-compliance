import logging
import os
import sys

# Silence MallocStackLogging warnings on macOS (must be set before torch loads)
os.environ["MallocStackLogging"] = "0"
os.environ["MallocStackLoggingNoCompact"] = "1"

from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Load environment variables from .env (or ENV_FILE if specified for testing) as early as possible
_env_file = os.getenv("ENV_FILE", ".env")
load_dotenv(_REPO_ROOT / _env_file, override=True)

from domain_profile import PROFILE

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to the default if missing or malformed."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning("Env %s=%r is not a valid integer, using default %s", name, os.getenv(name), default)
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to the default if missing or malformed."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        logger.warning("Env %s=%r is not a valid number, using default %s", name, os.getenv(name), default)
        return default


# --- Directories Configuration ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
# Frontend talks to the backend through the interface; make the repo root importable
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DOCS_DIR = os.getenv("DOCS_DIR", str(_REPO_ROOT / "data_docs"))

# --- Pipeline Thresholds ---
MAX_HISTORY_MESSAGES = _env_int("MAX_HISTORY_MESSAGES", 16)
MAX_HISTORY_TOKENS = _env_int("MAX_HISTORY_TOKENS", 25000)
COMPRESS_TOKEN_THRESHOLD = _env_int("COMPRESS_TOKEN_THRESHOLD", 85000)
TOKEN_GROWTH_FACTOR = _env_float("TOKEN_GROWTH_FACTOR", 0.9)
MAX_TOOL_CALLS = _env_int("MAX_TOOL_CALLS", 8)
MAX_ITERATIONS = _env_int("MAX_ITERATIONS", 5)
RECURSION_LIMIT = _env_int("RECURSION_LIMIT", 50)
MAX_OUTPUT_TOKENS = _env_int("MAX_OUTPUT_TOKENS", 16384)

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)

# --- RAG Backend Configuration ---
# Options: "vector_chunking" (default: neural embeddings + reranker) or "page_index" (vectorless, tree navigation)
RAG_BACKEND = os.getenv("RAG_BACKEND", "vector_chunking")

# --- Callback Logging Configuration ---
import datetime
import json
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

LLM_PROMPT_LOG_DIR = os.getenv("LLM_PROMPT_LOG_PATH", "")


class JSONLoggerCallback(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        if not LLM_PROMPT_LOG_DIR:
            return
        try:
            # Create the directory if it does not exist yet
            os.makedirs(LLM_PROMPT_LOG_DIR, exist_ok=True)

            # File name pattern YYYY-MM-DD_HH-MM-SS_ffffff.json; microseconds
            # avoid collisions when several nodes run in parallel
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            filename = f"{timestamp}.json"
            filepath = os.path.join(LLM_PROMPT_LOG_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                for message_list in messages:
                    log_data = []
                    for msg in message_list:
                        msg_dict = {"role": msg.type, "content": msg.content}
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            msg_dict["tool_calls"] = msg.tool_calls
                        log_data.append(msg_dict)
                    # Human-readable format (indent=2)
                    f.write(json.dumps(log_data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.error("Error logging LLM prompt to %s: %s", LLM_PROMPT_LOG_DIR, e)


# --- LLM Provider & 3-Stage Model Configuration ---
# Options: "ollama", "openai", "anthropic", "google", "zai"
# Supports per-request model routing (REQUEST_1, REQUEST_2, REQUEST_3) with
# automatic fallback to the provider's base model for 100% backward compatibility.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
# shortcut: in-memory LLM cache resets on restart, upgrade to SQLiteCache if persistence across restarts is needed
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


def configure_llm_cache(enabled: bool = LLM_CACHE_ENABLED) -> None:
    """Enable or disable the global LangChain in-memory LLM response cache (`LLM_CACHE_ENABLED`)."""
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(InMemoryCache() if enabled else None)


configure_llm_cache(LLM_CACHE_ENABLED)

_PROVIDER_DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "ollama": ("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "google": ("GOOGLE_MODEL", "gemini-2.5-flash"),
    "zai": ("ANTHROPIC_MODEL", "glm-4.7"),
}

_LLM_INSTANCE_CACHE: dict[tuple[str, str], Any] = {}


def resolve_model_name(stage: int | None = None, provider: str | None = None) -> str:
    """Resolve the model name for a specific request stage (1, 2, or 3) or base provider.

    Resolution priority for stage i in {1, 2, 3}:
      1. `<PROVIDER>_MODEL_REQUEST_<i>` (e.g. `OPENAI_MODEL_REQUEST_1`)
      2. `LLM_MODEL_REQUEST_<i>` (provider-agnostic stage override)
      3. `<PROVIDER>_MODEL` (legacy single-model env var)
      4. Built-in default model for `provider`

    Args:
        stage: Optional request tier (1 = keyword/retrieval plan, 2 = Python formula gate,
            3 = final audit synthesis). If None, resolves the base model.
        provider: Optional LLM provider override (defaults to `LLM_PROVIDER`).

    Returns:
        Resolved model identifier string.
    """
    active_provider = (provider or LLM_PROVIDER).strip().lower()
    if active_provider not in _PROVIDER_DEFAULT_MODELS:
        raise ValueError(f"Unsupported LLM_PROVIDER: {active_provider}")

    env_key, default_model = _PROVIDER_DEFAULT_MODELS[active_provider]
    base_model = (os.getenv(env_key) or "").strip() or default_model

    if stage in (1, 2, 3):
        stage_model = (
            (os.getenv(f"{env_key}_REQUEST_{stage}") or "").strip()
            or (os.getenv(f"LLM_MODEL_REQUEST_{stage}") or "").strip()
            or base_model
        )
    else:
        stage_model = base_model

    if active_provider == "zai":
        # Strip context-window suffixes injected by external tooling (e.g. "[1m]") —
        # the API only accepts plain model codes; the suffix causes a 400 error.
        stage_model = stage_model.split("[")[0].strip() or default_model

    return stage_model


def create_llm(model_name: str | None = None, provider: str | None = None) -> Any:
    """Instantiate or reuse a cached LangChain chat model for `provider` and `model_name`.

    When all 3 request stages resolve to the same model name, this cache returns
    the exact same Python object (`llm_request_1 is llm_request_2 is llm_request_3 is llm`),
    guaranteeing 100% backward-compatible behavior.

    Args:
        model_name: Target model name. If None, resolves the base model for `provider`.
        provider: Target LLM provider. If None, uses `LLM_PROVIDER`.

    Returns:
        Configured LangChain chat model instance with streaming enabled.
    """
    active_provider = (provider or LLM_PROVIDER).strip().lower()
    resolved_model = (model_name or "").strip() or resolve_model_name(stage=None, provider=active_provider)
    if active_provider == "zai":
        resolved_model = resolved_model.split("[")[0].strip() or "glm-4.7"

    cache_key = (active_provider, resolved_model)
    if cache_key in _LLM_INSTANCE_CACHE:
        return _LLM_INSTANCE_CACHE[cache_key]

    if active_provider == "ollama":
        from langchain_ollama import ChatOllama
        seed_env = os.getenv("OLLAMA_SEED") or os.getenv("OPENAI_SEED")
        ollama_seed: int | None = None
        if seed_env is not None and seed_env.strip().lower() not in ("none", "null", ""):
            try:
                ollama_seed = int(seed_env.strip())
            except ValueError:
                ollama_seed = None
        instance = ChatOllama(model=resolved_model, temperature=0, seed=ollama_seed, streaming=True)
    elif active_provider == "openai":
        from langchain_openai import ChatOpenAI
        # Reasoning models (e.g. o1, o3, *-sol) do not support temperature (must be None)
        is_reasoning_model = any(
            kw in resolved_model.lower() for kw in ("-sol", "o1", "o3", "reasoning")
        )
        if is_reasoning_model:
            openai_temp: float | None = None
        else:
            temp_env = os.getenv("OPENAI_TEMPERATURE")
            if temp_env is not None and temp_env.strip().lower() not in ("none", "null", ""):
                openai_temp = float(temp_env)
            else:
                openai_temp = 0.0

        seed_env = os.getenv("OPENAI_SEED")
        openai_seed: int | None = None
        if seed_env is not None and seed_env.strip().lower() not in ("none", "null", ""):
            try:
                openai_seed = int(seed_env.strip())
            except ValueError:
                openai_seed = None

        instance = ChatOpenAI(
            model=resolved_model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=openai_temp,
            seed=openai_seed,
            streaming=True,
        )
    elif active_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        instance = ChatAnthropic(
            model=resolved_model,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            streaming=True,
        )
    elif active_provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "LLM_PROVIDER=google requires the optional Google provider package. "
                "Install it with: pip install langchain-google-genai "
                "(it is not in requirements.txt because its pydantic requirement "
                "conflicts with gradio 5.x)."
            ) from e
        instance = ChatGoogleGenerativeAI(
            model=resolved_model,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            streaming=True,
        )
    elif active_provider == "zai":
        from langchain_anthropic import ChatAnthropic
        zai_key = os.getenv("ANTHROPIC_AUTH_TOKEN")
        zai_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.z.ai/api/anthropic")
        instance = ChatAnthropic(
            model=resolved_model,
            api_key=zai_key,
            base_url=zai_url,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0,
            streaming=True,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {active_provider}")

    if LLM_PROMPT_LOG_DIR:
        # Assign to the callbacks attribute instead of with_config(): the config
        # would be lost across llm.bind_tools() / llm.with_structured_output()
        instance.callbacks = [JSONLoggerCallback()]

    _LLM_INSTANCE_CACHE[cache_key] = instance
    return instance


MODEL_REQUEST_1 = resolve_model_name(stage=1)
MODEL_REQUEST_2 = resolve_model_name(stage=2)
MODEL_REQUEST_3 = resolve_model_name(stage=3)

llm_request_1 = create_llm(MODEL_REQUEST_1)
llm_request_2 = create_llm(MODEL_REQUEST_2)
llm_request_3 = create_llm(MODEL_REQUEST_3)
llm = llm_request_3


import unicodedata

# --- UI Configuration ---
SOURCE_NAMES = {}

# Map profile-declared display names, normalized in both NFC and NFD
for k, v in PROFILE["source_display_names"].items():
    v_clean = unicodedata.normalize('NFC', v)
    SOURCE_NAMES[unicodedata.normalize('NFC', k)] = v_clean
    SOURCE_NAMES[unicodedata.normalize('NFD', k)] = v_clean

# Dynamically populate display names from the docs folder (without overwriting profile display names)
docs_path = Path(DOCS_DIR)
if docs_path.exists():
    for f in docs_path.iterdir():
        if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".md"]:
            # Clean name: replace underscores/dashes and format nicely
            clean_name = f.stem.replace("_", " ").replace("-", " ")
            if clean_name.islower():
                clean_name = clean_name.title()

            v_clean = unicodedata.normalize('NFC', clean_name)

            name_nfc = unicodedata.normalize('NFC', f.name)
            name_nfd = unicodedata.normalize('NFD', f.name)

            # Add mappings for both NFC/NFD keys only if not already declared in profile.json
            SOURCE_NAMES.setdefault(name_nfc, v_clean)
            SOURCE_NAMES.setdefault(name_nfd, v_clean)

# --- Web Server Configuration ---
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = _env_int("GRADIO_SERVER_PORT", 7860)

# --- Retrieval Configuration ---
# Default number of parent documents returned by the search_documents tool
DEFAULT_SEARCH_LIMIT = _env_int("DEFAULT_SEARCH_LIMIT", 5)
