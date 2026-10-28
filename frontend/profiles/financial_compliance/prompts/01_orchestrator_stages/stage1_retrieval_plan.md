[PHASE 1 — RETRIEVAL & AUDIT DISPATCH]
1. Call `{tool_name_to_force}` first (`run_nav_audit` in parallel ONLY for NAV pricing or dated single-class fee audits). Defer `calculate_decimal` to Phase 2.
2. QUERY FIDELITY: Strip narrative/IT filler; keep ALL statutory clauses, tables, points, entities, breach qualifiers, and sub-questions in `query`.
3. CLARIFICATION BIAS & TEMPORAL FILTER: Act without clarification pauses on clear audit queries. Pass `audit_date` ONLY for single-date queries; omit `audit_date` when comparing regimes/years (predecessor vs. successor circulars) or undated.
4. Optional: 1-sentence intent under `{reasoning_primary}`. Never emit `{conclusion_primary}` in Phase 1.
