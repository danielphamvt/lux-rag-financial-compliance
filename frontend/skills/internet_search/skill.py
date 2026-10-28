import json
import logging
import os
import subprocess
import sys

from langchain_core.tools import tool

from domain_profile import PROFILE

logger = logging.getLogger(__name__)


# Read env directly instead of importing config: this skill must stay
# runnable standalone without triggering the LLM/embeddings/Qdrant
# initialization inside config.py.
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


SEARCH_TIMEOUT_SECONDS = _int_env("SEARCH_TIMEOUT_SECONDS", 30)
SEARCH_MAX_RESULTS = _int_env("SEARCH_MAX_RESULTS", 3)

# The search runs in a short-lived subprocess instead of in-process:
# search engines rate-limit per interpreter session, and running it outside
# the app/event loop avoids cross-request cache pollution. The query, the
# result cap and the language-specific query suffixes are passed via argv;
# the suffix list is domain-profile configuration.
_SEARCH_SCRIPT = """
import json
import sys
import re
import warnings
warnings.filterwarnings("ignore")

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

raw_query = sys.argv[1].strip() if len(sys.argv) > 1 else ""

# Result cap and query suffixes are passed by the parent process via argv
try:
    max_results = max(1, int(sys.argv[2]))
except (IndexError, ValueError):
    max_results = 3
try:
    suffixes = json.loads(sys.argv[3])
except (IndexError, ValueError):
    suffixes = []

# Strip trailing punctuation
cleaned = re.sub(r"[\\?\\.!\\*,;:_]+$", "", raw_query).strip()
# Strip trailing question filler words (language-specific, from the profile)
if suffixes:
    pattern = r"\\s+(" + "|".join(re.escape(s) for s in suffixes) + ")$"
    cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

candidates = [cleaned] if cleaned else [raw_query]
if raw_query and raw_query != cleaned:
    candidates.append(raw_query)

results = []
try:
    with DDGS() as ddgs:
        for q in candidates:
            for b in ["auto", "html", "lite"]:
                try:
                    res = list(ddgs.text(q, backend=b, max_results=max_results))
                    if res:
                        results = res
                        break
                except Exception:
                    continue
            if results:
                break
    print(json.dumps(results))
except Exception as exc:
    sys.stderr.write(f"{type(exc).__name__}: {exc}\\n")
    sys.exit(1)
"""


@tool
def internet_search(query: str) -> str:
    """
    Search the internet for up-to-date information, news, or articles.
    Use this tool ONLY when the user's question is about recent events,
    recently updated documents not in the database, or real-world situations.

    Args:
        query: The search query string. Should include the most relevant keywords.
    """
    strings = PROFILE["strings"]
    try:
        # Strip stray quotes the user may have pasted in: quotes force exact
        # phrase matching in most search engines and often return nothing.
        safe_query = query.strip().strip('"').strip("'").strip()

        # Query, result cap and suffix list are passed via argv — they are
        # never interpolated into the subprocess source, so their content
        # cannot break out of or inject code. (Args after "-c <code>" go
        # straight into sys.argv, without a shell.)
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _SEARCH_SCRIPT,
                safe_query,
                str(SEARCH_MAX_RESULTS),
                json.dumps(PROFILE["internet_search"]["query_suffixes"]),
            ],
            capture_output=True,
            text=True,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )

        if process.returncode != 0:
            err_detail = (process.stderr or "").strip() or f"exit code {process.returncode}"
            logger.warning("Internet search subprocess failed: %s", err_detail)
            return strings["internet_error"].format(error=err_detail)

        try:
            results = json.loads(process.stdout.strip() or "[]")
        except json.JSONDecodeError as exc:
            logger.warning("Internet search returned invalid JSON: %s", exc)
            return strings["internet_error"].format(error="JSONDecodeError")

        if not results:
            return strings["internet_no_results"]

        formatted_results = [
            f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}"
            for r in results
        ]
        result_text = "\n\n".join(formatted_results)

        # Add a prefix to the result to help the LLM recognize this is internet data
        return f"[INTERNET_DATA_START]\n{result_text}\n[INTERNET_DATA_END]"
    except subprocess.TimeoutExpired:
        return strings["internet_timeout"]
    except Exception as e:
        return strings["internet_error"].format(error=type(e).__name__)
