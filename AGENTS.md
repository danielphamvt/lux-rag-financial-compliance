# AGENTS.md

This file provides guidance to any AI Agent when working with code in this repository.

## Role
Your role is to analyze user requirements, delegate tasks to appropriate sub-agents, and ensure cohesive delivery of features that meet specifications and architectural standards.
**IMPORTANT:** Analyze the skills catalog and activate the skills that are needed for the task during the process.

## Principles (always apply)

- **YAGNI** (You Aren't Gonna Need It) — **KISS** (Keep It Simple, Stupid) — **DRY** (Don't Repeat Yourself)
- **Explainable, Human Readable and Maintainable**: Write code that is easy to understand, modify, and maintain. Avoid unnecessary abstractions and complex solutions.
- **No Hardcoding & No Special-Casing**: Never hardcode values, create scenario-specific shortcuts, or inject synthetic harness crutches to pass evaluations; always resolve reasoning gaps at the root through generalized domain principles and prompt governance.
- When working on a specific module, read its relevant documentation or `README.md` before planning or implementing anything
- Whenever architecture, public APIs, or configuration changes meaningfully, update `README.md`
- Concise reports; list unresolved questions at end

## Running Environment

- Python: always use virtual environment `.venv`

## Essential Commands

- **Tests (Software & Frontend)**: `make test` (or `pytest`) — runs all technical & frontend suites in `tests/` on every code change
- **Evaluation (On-Demand)**: `make eval` (or `pytest evaluation/`) — runs paper metric evaluations (`eval_*.py`) only when requested
- **Evaluation Summary Dashboard**: `make summary` (or `make eval-summary`) — print detailed evaluation summary of 4 types (Accuracy, Latency, Metrics)
- **Benchmark (Full Research)**: `make benchmark` (or `python -m evaluation.run_benchmark --backend vector_chunking|page_index`)
- **UI**: `python frontend/app.py` (port 7860)
- **Make shortcuts**: `make ingest | run | test | eval | summary | benchmark | stop | restart | clean`

## Core Architecture & Invariants

1. **Evaluation results are the golden reference:** Never alter the eval chain (`rag_engine/backends/<backend>/` + `evaluation/`) without re-running the benchmark, diffing `benchmark_results.json` (all metrics identical, latency excluded), and passing all tests.
2. **One implementation, two consumers:** UI and evaluation use the identical backend via `rag_engine/core/interface.py`.
3. **`retrieve()` is deterministic:** Same index + query + `audit_date` → same result. No LLM calls or hidden state in retrieval; conversational logic stays in the frontend.
4. **Namespace isolation:** `vector_chunking` owns `data/compliance_rag/`; `page_index` and future backends get dedicated storage under `data/indexes/<backend>/`. Never share or overwrite indexes.
5. **Adding backends:** Create `rag_engine/backends/<name>/` implementing `ingest()` and `retrieve()`, register in `rag_engine/core/registry.py`, and pass conformance tests. Never mutate existing backends (no architecture branches).
6. **Git etiquette:** Move files using `git mv`. Use conventional commits.

## Code Standards

- No hardcoded scientific constants — all params should be in config file
- Never silently swallow exceptions or switch to fallback — catch the error and report it
- Type hints on all public functions; docstrings with Args/Returns

## System Prompts & Frontend Governance

- Reference guide: `frontend/profiles/financial_compliance/prompts/README.md`.
- Edit prompts only in `prompts/*.md` and `_partials/_*.md`.
- Principle-driven: Never hardcode scenario-specific benchmark figures, answers, or section numbers into prompts or tools.

## [IMPORTANT] Consider Modularization

- If a code file exceeds 350 lines of code, consider modularizing it
- Check existing modules before creating new
- Analyze logical separation boundaries (functions, classes, concerns)
- Use snake_case for Python modules (PEP 8) and kebab-case for Markdown prompts/docs with long descriptive names so they are self-documenting for LLM tools (Grep, Glob, Search)
- Write descriptive code comments
- After modularization, continue with main task
- When not to modularize: Markdown files, plain text files, bash scripts, configuration files, environment variables files, etc.
