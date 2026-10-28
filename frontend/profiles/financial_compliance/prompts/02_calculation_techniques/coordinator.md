- **Zero Mental Math & Structured `calculate_decimal` Execution**:
  - Pass every quantitative step to `calculate_decimal` as `Label = expression` (one formula per line, expanding shorthand such as `M` → `,000,000`).
  - Expand intermediate fractions inline rather than multiplying rounded 4-decimal intermediates.
  - Organize multi-step calculations across three logical layers:
    1. **Individual Legs & Marginal Tier Partition**: Compute each position, share class, expense category, or marginal bracket on its own line. On tiered fee schedules, cap each bounded bracket at its bracket width (`upper_cap - lower_cap`) and apply the top rate only to the excess (`gav - prior_upper_cap`).
    2. **Homogeneous Units & Predicate-First Filtering**: Sum or compare only identical units and post-trade states (`%` with `%`, same currency with same currency). For threshold baskets, include only legs satisfying the statutory predicate.
    3. **Economic Sign & Cost Symmetry**: Compute variances (`booked - replicated` or `actual - ceiling`) and unit conversions (`%`, `bps`, currency) as explicit lines in `calculate_decimal` rather than mental subtraction in prose. If citing an excess in both `%` (or pp) and currency, execute both lines in `calculate_decimal`.

## TECHNIQUE ACTIVATION TABLE

This coordinator table maps query and retrieved-context keywords to specialized domain calculation techniques in this folder. At Phase 2 synthesis, matching techniques are automatically activated and injected into `{active_calculation_techniques}`.

| Technique File | Trigger Keywords |
| :--- | :--- |
| `breach-indemnification-accounting-vs-economic.md` | `indemnif`, `accounting method`, `economic method`, `realized loss`, `unrealized loss`, `transaction cost`, `cost method`, `fifo`, `lifo`, `weighted average` |
| `swing-pricing-and-anti-dilution.md` | `swing`, `swung`, `unswung`, `anti-dilution`, `dilution`, `liquidity levy`, `net capital flow` |
| `ucits-concentration-and-exposure-caps.md` | `concentration`, `single-issuer`, `single issuer`, `covered bond`, `corporate group`, `fund of funds`, `non-ucits`, `commitment approach`, `global exposure`, `borrowing`, `article 51`, `article 52`, `article 55`, `5/10/40`, `basket`, `ceiling`, `aggregate limit`, `headroom` |
| `fee-accrual-and-tiered-schedules.md` | `management fee`, `taxe d'abonnement`, `subscription tax`, `ongoing charges`, `day-count`, `actual/365`, `actual/360`, `actual/366`, `tiered`, `tier 1`, `tier 2`, `breakpoint`, `statutory interest`, `fx`, `hedged` |
| `nav-error-and-hwm-performance-fee.md` | `replicated net assets`, `pricing error`, `tolerance threshold`, `high-water mark`, `hwm`, `performance fee`, `ex-dividend`, `dividend` |
