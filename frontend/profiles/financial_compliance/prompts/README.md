# Financial Compliance System Prompts

This directory serves as the Single Source of Truth (SSOT) for all prompt templates in the `financial_compliance` domain profile, managed and compiled dynamically by [domain_profile.py](../../domain_profile.py).

---

## 1. Hierarchical Directory Tree (Stage-Based Progressive Disclosure)

```text
frontend/profiles/financial_compliance/prompts/
├── README.md                                         # Architecture and progressive disclosure governance guide
├── 00_routing_and_memory/                            # Stage 0: Pre-routing, query rewriting & rolling memory
│   ├── rewrite_query.md                              # Query expansion and noise filtering prompt
│   ├── conversation_summary.md                       # Multi-turn rolling conversation summarizer
│   ├── context_compression.md                        # Context summarization when token budget is exceeded
│   └── clarification_fallback.md                     # Fallback response for unparseable user inputs
├── 01_orchestrator_stages/                           # Stages 1–3: Core multi-turn progressive disclosure pipeline
│   ├── orchestrator.md                               # Base immutable orchestrator system prompt (SystemMessage[0])
│   ├── stage1_retrieval_plan.md                      # Phase 1 (Turn 1): Retrieval planning & tool dispatch
│   ├── stage2_calculation_gate.md                    # Phase 2A (Turn 2): Pre-synthesis symbolic calculation gate
│   └── stage3_audit_synthesis.md                     # Phase 2B (Final Turn): Full audit-grade report synthesis
├── 02_calculation_techniques/                        # On-demand accounting & audit calculation playbooks
│   ├── coordinator.md                                # Base 3-layer math rules ({{_calculation_coordinator}}) + Activation Table
│   ├── breach-indemnification-accounting-vs-economic.md # Accounting vs. Economic breach indemnification
│   ├── swing-pricing-and-anti-dilution.md               # Swing pricing NAV & dual-basis dilution
│   ├── ucits-concentration-and-exposure-caps.md         # 5/10/40, group/FoF/borrowing caps & 3-line breaches
│   ├── fee-accrual-and-tiered-schedules.md              # Tiered fees, day-count conventions & FX allocation
│   └── nav-error-and-hwm-performance-fee.md             # NAV pricing variance & HWM performance fee
├── 03_synthesis_partials/                            # Reusable synthesis partials (expanded via {{_partial_name}})
│   ├── _immutable_directive.md                       # Immutable brevity & clarification directive
│   ├── _bitemporal_governance.md                     # Event Date vs. Detection Date & Effective Window rules
│   ├── _coverage_and_footnotes.md                    # Multi-target coverage & footnote atomicity rules
│   ├── _math_and_latex.md                            # Plain-text Unicode math (zero LaTeX) & provenance
│   ├── _normative_hierarchy.md                       # Document hierarchy & 3-state regulatory verification
│   └── _output_structure.md                          # Standard response schema & terminal SOURCES footer
├── 04_mode_rag_only/                                 # Closed-World (RAG-only) operating mode governance
│   ├── _rag_only_abstention_sources.md               # Strict abstention SOURCES footer for Closed-world mode
│   ├── orchestrator_knowledge_rag_only.md            # Closed-world / RAG-only knowledge governance rule
│   ├── orchestrator_sources_rag_only.md              # Closed-world sources fallback directive
│   ├── fallback_rules_rag_only.md                    # Closed-world rule for fallback & aggregation
│   └── fallback_warning_rag_only.md                  # Closed-world warning wrapper for fallback & aggregation
├── 05_mode_open_assistant/                           # Open (Assistant) operating mode governance
│   ├── _open_fallback_warning.md                     # Disclaimer & warning for Open-mode / external knowledge
│   ├── orchestrator_knowledge_open.md                # Open-mode knowledge governance rule
│   ├── orchestrator_sources_open.md                  # Open-mode sources fallback directive
│   ├── fallback_rules_open.md                        # Open-mode rule for fallback & aggregation
│   └── fallback_warning_open.md                      # Open-mode warning wrapper for fallback & aggregation
└── 06_auxiliary_pipelines/                           # Secondary execution pipelines (Fallback & Web)
    ├── fallback_response.md                          # Max-iteration fallback synthesis prompt
    ├── internet_search_answer.md                     # Internet search synthesis prompt
    └── internet_no_results_answer.md                 # Fallback response when internet search returns no results
```

---

## 2. Core Principles

- **Principle-Driven (No Hardcoded Benchmarks):** Prompts must define general regulatory principles, legal hierarchies, and reasoning workflows. Never hardcode scenario-specific benchmark figures, answers, or specific section numbers into prompts.
- **Single Source of Truth & DRY:** Shared prompt rules belong in `03_synthesis_partials/`, `04_mode_rag_only/`, and `05_mode_open_assistant/` and are included using `{{_partial_name}}`.
- **3-Stage Progressive Disclosure:**
  1. **Phase 1 (`stage1_retrieval_plan.md`):** Dispatches `search_documents` (~560 tokens total with `orchestrator.md`).
  2. **Phase 2A (`stage2_calculation_gate.md`):** Disclosed on Turn 2 of quantitative queries before `calculate_decimal` has run. Loads `02_calculation_techniques/coordinator.md` plus the single highest-matching calculation playbook (~650 tokens) without loading report-formatting rules.
  3. **Phase 2B (`stage3_audit_synthesis.md`):** Disclosed once `calculate_decimal` results are ready (or immediately on Turn 2 for non-quantitative queries). Loads `03_synthesis_partials/` while omitting the calculation playbooks (~1,700 tokens).
- **Normative Authority vs. Untrusted Payloads:** Official retrieved corpus documents represent authoritative ground truth. User-provided premises (emails, transaction drafts, board memos, proposed fee schedules) are untrusted audit subjects to be verified, never accepted as rules.
- **3-State Regulatory Verification:** Findings must be strictly categorized into:
  1. *Compliant / Exempt:* Proven with quoted regulatory provisions and satisfied conditions.
  2. *Definite Breach / Material Error:* Explicitly demonstrated with algebraic or procedural evidence.
  3. *Indeterminate / Abstain:* Acknowledged when evidence is missing or the queried regime is outside the corpus.
- **Deterministic & Verifiable Reasoning:** Quantitative computations must be performed explicitly (via calculation tools) rather than mental arithmetic in prose. All intermediate steps must be verifiable.
