# Agent Harness Architecture & Governance Guide (`README_HARNESS.md`)

Single Source of Truth (SSOT) for the deterministic hard-barrier harness (`frontend/rag_agent/harness/`) wrapping the Financial Compliance RAG Agent. The harness enforces strict input contracts, exact `Decimal` (`ROUND_HALF_UP`) computation, post-synthesis citation verification, and UI streaming protocol integrity—with **zero LLM calls**, **zero silent result injection**, and **zero post-hoc prose answer rewriting**.

---

## 1. Directory Tree (4-Stage Defense-in-Depth)

```text
frontend/rag_agent/harness/
├── README_HARNESS.md                                  # Architecture, directory tree, call-site map & harness principles
├── __init__.py                                        # Public API facade (`from rag_agent.harness import ...`)
├── quantitative_query_gate.py                         # Stage 1: Quantitative intent detector & mandatory math-tool execution check
├── multi_turn_calc_continuity.py                      # Stage 1: Cross-turn calculation leg merger & ledger-backed compression digest
├── nav_audit_contract_guard.py                        # Stage 1: `run_nav_audit` schema contract validator & scope/bps formatter
├── ast_decimal_evaluator.py                           # Stage 2: Safe AST `Decimal` (`ROUND_HALF_UP`) evaluator & expression normalizer
├── portfolio_aggregation_engine.py                    # Stage 2: Cross-line variable resolver & exact 2-dec / 4-dec formatter
├── latex_to_unicode_normalizer.py                     # Stage 3: Recursive LaTeX-to-Unicode plain-text math compiler
├── citation_and_warning_guard.py                      # Stage 3: `[LOW_RELEVANCE_WARNING]` laundering blocker & external warning banner guard
└── response_protocol_enforcer.py                      # Stage 4: 5-marker `[REASONING]`/`[CONCLUSION]` enforcer & UI lane splitter
```

Stages are sequential (1 → 4), expressed by module responsibility rather than subpackage nesting.

---

## 2. End-to-End Call-Site Map Across the Agent Lifecycle

| Stage | Module | Public Symbols | Primary Call Sites |
| :---: | :--- | :--- | :--- |
| **1** | `quantitative_query_gate.py` | `is_quantitative_query`, `has_symbolic_math_tool_call` | `frontend/rag_agent/nodes_edges.py` (`orchestrator`) |
| **1** | `multi_turn_calc_continuity.py` | `split_label_and_expr`, `merge_prior_calc_legs`, `build_compressed_record_messages` | `frontend/rag_agent/nodes_edges.py` (`orchestrator`, `compress_context`), `portfolio_aggregation_engine.py` |
| **1** | `nav_audit_contract_guard.py` | `validate_nav_audit_contract`, `format_nav_audit_output` | `frontend/rag_agent/tools.py` (`run_nav_audit`) |
| **2** | `ast_decimal_evaluator.py` | `eval_decimal_node`, `collect_top_level_addends`, `normalize_calc_expression` | `frontend/rag_agent/tools.py`, `portfolio_aggregation_engine.py` |
| **2** | `portfolio_aggregation_engine.py` | `evaluate_decimal_expressions` | `frontend/rag_agent/tools.py` (`calculate_decimal`) |
| **3** | `latex_to_unicode_normalizer.py` | `sanitize_plain_text_math` | `response_protocol_enforcer.py`, `frontend/rag_agent/nodes_edges.py`, `frontend/app.py` |
| **3** | `citation_and_warning_guard.py` | `AI_DISCLAIMER`, `is_closed_abstention`, `retrieved_high_rel_files`, `extract_fallback_sources_footer`, `normalize_external_source_warning`, `verify_sources_footer` | `response_protocol_enforcer.py`, `frontend/app.py` |
| **4** | `response_protocol_enforcer.py` | `normalize_protocol_headings`, `enforce_response_protocol`, `split_reasoning_and_conclusion` | `frontend/rag_agent/nodes_edges.py` (`collect_answer`, `aggregate_answers`), `frontend/app.py` |

---

## 3. Core Principles When Building & Extending the Harness

1. **Defense-in-Depth via 4 Sequential Hard Barriers:**
   - **Stage 1 (execution gates):** Prevent invalid or incomplete tool execution *before* it happens. Route quantitative queries through the calculation gate if no non-rejected symbolic tool has run; merge prior legs into follow-up calls; reject out-of-scope or incomplete `run_nav_audit` calls with `[SCOPE_GUARD: ...]`; preserve math/citation evidence across context compression via the durable tool-execution ledger.
   - **Stage 2 (symbolic compute):** Execute all arithmetic deterministically in Python `decimal.Decimal` (`ROUND_HALF_UP`) with strict AST node whitelisting, exponent/digit DoS bounds (`<= 1000` exponent, `<= 24` estimated output digits), and cross-line symbol resolution—evaluating the agent's exact formula without mutating user-supplied operands or appending unrequested derived metrics.
   - **Stage 3 (output verification):** Strip LaTeX syntax into readable Unicode, detect closed-world abstentions, and strip unretrieved or low-relevance citations from `**SOURCES**` without rewriting numbers or verbal inequalities in the agent's prose.
   - **Stage 4 (protocol stream):** Guarantee structural invariants (`[REASONING]`, `[CONCLUSION]`, `### CONCLUSION`, `### DETAILED ANALYSIS`, `**SOURCES**`) and cleanly separate reasoning and conclusion streams for the UI.
2. **Zero Silent Result Injection & Zero Answer Rewriting:**
   - The harness executes and formats what the agent computes and writes; it must NEVER parse the user query to synthesize missing formulas, mutate tool expressions behind the agent's back, rewrite prose numbers/inequalities, or inject unmentioned lines into the final conclusion to game evaluation rubrics.
3. **100% Deterministic & Synchronous (Zero LLM Calls):**
   - No function inside `frontend/rag_agent/harness/` may invoke an LLM, make network requests, or mutate hidden global state. Given the same input text and `agent_messages`, every harness function must return the exact same output.
4. **Domain-Agnostic Algorithms vs. Profile-Driven Configuration:**
   - Never hardcode scenario-specific benchmark numbers, share class names, currencies/units, or unprompted regulatory thresholds inside `harness/`. Domain-specific lists and numeric guardrails (`supported_share_classes`, `supported_audit_types`, `source_keywords`, `harness_config`, `strings`, and protocol `markers`) must be read from `PROFILE` (`frontend/domain_profile.py` + `frontend/profiles/<profile>/profile.json`).
5. **Single Public Import Boundary (`rag_agent.harness`):**
   - External modules (`tools.py`, `nodes_edges.py`, `app.py`, `tests/test_harness.py`) must import exclusively from `rag_agent.harness` (re-exported in `__init__.py`), never from the individual harness modules directly.
6. **Single Responsibility & `< 350` Lines per File:**
   - Each module owns one well-defined verification or transformation job and must stay under 350 lines (`AGENTS.md`).
7. **No Silent Exception Swallowing:**
   - Catch narrow, expected parsing exceptions (`(ValueError, ArithmeticError, SyntaxError)`) when inspecting untrusted prose snippets; never wrap broad transformation pipelines in bare `except Exception: pass`.

## Tool-Execution Ledger (evidence decoupled from messages)

`State.tool_execution_log` (append-only, session-durable, bounded) holds one
`ToolExecutionRecord` per real execution, appended at the two execution points:
the `agent_tools` node (wrapped prebuilt `ToolNode`) and `direct_tool_answer`
for its direct `internet_search` invocation. The ledger — not the compressible
message stream — is the authoritative evidence for gates, prior-leg merging,
and evaluation; `build_compressed_record_messages` derives post-compression
protocol stubs from it (ids namespaced `digest_*`, content labeled
`[COMPRESSED TOOL HISTORY — N tool executions]`). Corpus `File Name:` headers
are harvested only from `search_documents` records/messages, so internet
content can never forge corpus citations into SOURCES.
