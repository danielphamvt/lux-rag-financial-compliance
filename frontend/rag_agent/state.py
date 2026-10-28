from dataclasses import dataclass, field
import hashlib
from typing import Annotated, Any

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def accumulate_or_reset(existing: list[dict], new: list[dict]) -> list[dict]:
    """Append `new` items to `existing`, or clear the list when a `{'__reset__': True}` sentinel is present.

    Args:
        existing: Previously accumulated list of answer dicts.
        new: Incoming list of answer dicts or reset sentinel.

    Returns:
        Updated list of answer dicts.
    """
    if new and any(isinstance(item, dict) and item.get('__reset__') for item in new):
        return []
    return existing + new


@dataclass
class ToolExecutionRecord:
    """Durable record of one real tool execution, appended at the execution point.

    The ledger is the authoritative evidence source for gates and evaluation.
    """

    tool_call_id: str = ""
    tool_name: str = ""
    args: dict = field(default_factory=dict)
    output: str = ""
    is_recent: bool = True
    output_sha256: str = ""
    output_first_line: str = ""


# ponytail: bounded ledger — the last LEDGER_FULL_RECORD_LIMIT records keep full output;
# older ones are truncated to hash + first line. Raise the limit (or persist the ledger)
# if long sessions need full fidelity beyond the cap.
LEDGER_FULL_RECORD_LIMIT = 24


def _demote_record(r: ToolExecutionRecord) -> ToolExecutionRecord:
    """Drop full output, keeping the hash + first line for auditability."""
    if not r.output:
        return r
    sha = r.output_sha256 or hashlib.sha256(r.output.encode("utf-8")).hexdigest()
    first_line = r.output_first_line or (r.output.splitlines()[0] if r.output else "")
    return ToolExecutionRecord(
        tool_name=r.tool_name,
        tool_call_id=r.tool_call_id,
        args=dict(r.args),
        output="",
        is_recent=False,
        output_sha256=sha,
        output_first_line=first_line,
    )


def append_tool_records(existing: list, new: list) -> list:
    """Append tool-execution records; session-durable (never reset by _turn_reset).

    Bounds growth by demoting records beyond the full-output limit to hash + first line.

    Args:
        existing: Previously recorded `ToolExecutionRecord` entries.
        new: Newly emitted `ToolExecutionRecord` entries to append.

    Returns:
        Bounded list of `ToolExecutionRecord` entries.
    """
    merged = existing + [r for r in (new or []) if r is not None]
    cutoff = len(merged) - LEDGER_FULL_RECORD_LIMIT
    if cutoff <= 0:
        return merged
    return [(_demote_record(r) if isinstance(r, ToolExecutionRecord) else r) if idx < cutoff else r for idx, r in enumerate(merged)]


def set_union_or_reset(a: set[str], b: set[str] | dict) -> set[str]:
    """Union two string sets, or clear the set when a `{'__reset__': True}` sentinel is received.

    Args:
        a: Existing set of retrieval keys.
        b: Incoming set of retrieval keys or reset sentinel dict.

    Returns:
        Updated set of retrieval keys.
    """
    if isinstance(b, dict) and b.get("__reset__"):
        return set()
    if isinstance(b, set):
        return a | b
    return a


def add_or_reset(a: int, b: int | dict) -> int:
    """Add integer counter `b` to `a`, or reset to `0` when a `{'__reset__': True}` sentinel is received.

    Args:
        a: Existing counter value.
        b: Increment integer or reset sentinel dict.

    Returns:
        Updated counter value.
    """
    if isinstance(b, dict) and b.get("__reset__"):
        return 0
    if isinstance(b, int):
        return a + b
    return a


def pick_last(a: Any, b: Any) -> Any:
    """Reducer that keeps the latest value `b` for concurrent branches returning the same key.

    Args:
        a: Previous state value.
        b: Latest state value.

    Returns:
        The latest value `b`.
    """
    return b

class State(MessagesState):
    questionIsClear: bool = False
    conversation_summary: str = ""
    originalQuery: str = ""
    rewrittenQuestions: list[Any] = []
    agent_answers: Annotated[list[dict], accumulate_or_reset] = []
    # Agent-loop keys with pick_last / *_or_reset reducers so parallel
    # sub-query branches do not clobber each other on shared keys.
    question: Annotated[str, pick_last] = ""
    question_index: Annotated[int, pick_last] = 0
    context_summary: Annotated[str, pick_last] = ""
    tool_call_count: Annotated[int, add_or_reset] = 0
    iteration_count: Annotated[int, add_or_reset] = 0
    retrieval_keys: Annotated[set[str], set_union_or_reset] = set()
    agent_messages: Annotated[list, add_messages] = []
    tool_execution_log: Annotated[list, append_tool_records] = []
    forced_tool: Annotated[str | None, pick_last] = None
    source_mode: Annotated[str, pick_last] = "RAG only"
    history_context: Annotated[str, pick_last] = ""

# The agent-loop nodes (orchestrator/tools/...) run directly on this shared
# state and use the private `agent_messages` channel: their internal reasoning
# (tool calls, tool results, draft answers) stays out of the user-visible
# `messages`. Only `agent_answers` crosses back into the parent flow
# (written by collect_answer). The pick_last / *_or_reset reducers keep
# parallel sub-query branches from clobbering each other on shared keys.

class SubQuery(BaseModel):
    query: str = Field(description="The rewritten, self-contained question.")
    forced_tool: str | None = Field(default=None, description="Tool forced by a slash command, if any.")

class QueryAnalysis(BaseModel):
    is_clear: bool = Field(default=True, description="Defaults to true (the question is clear). Return false only if the question is completely meaningless.")
    is_out_of_domain: bool = Field(
        default=False,
        description="Set to true ONLY if the user query is completely unrelated to finance, law, regulations, fund accounting, or compliance auditing (e.g., cooking, sports, weather, creative writing).",
    )
    questions: list[SubQuery] = Field(description="List of rewritten questions.")
    clarification_needed: str | None = Field(default="", description="If is_clear=false, explain why here. Otherwise leave empty.")
