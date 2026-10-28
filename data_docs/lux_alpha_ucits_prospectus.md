# LUX ALPHA SICAV — UCITS PROSPECTUS (EXTRACT: VALUATION & FEE SCHEDULE)
**Jurisdiction:** Grand Duchy of Luxembourg (Part I of the Law of 17 December 2010 on UCITS)  
**Regulatory Oversight:** Commission de Surveillance du Secteur Financier (CSSF)  
**Depositary Bank:** LuxTrust Depositary Bank S.A. (pursuant to Article 22 of Directive 2009/65/EC)  
**Fund Administrator:** Alpha Fund Administration Services S.A.  
**Effective Date:** `2023-01-01` to `9999-12-31` (Version 2.4)

---

## Section 8. Determination of the Net Asset Value (NAV)

### 8.1. Valuation Point and Pricing Rules
The Net Asset Value per Share of each Share Class of **Lux Alpha SICAV** is calculated on each Luxembourg Valuation Day at **16:00 CET** based on the last available official closing price on the primary stock exchange.
* If a security's market price is stale (older than 1 trading day) or unrepresentative, fair value adjustments must be approved by the Board of Directors.
* Cash dividends declared on equity holdings must be accrued in the fund's receivables on the **Ex-Dividend Date (Ex-Date)** prior to deducting management fees.
* The NAV per Share is rounded to **two (2) decimal places** using standard commercial half-up rounding (`ROUND_HALF_UP`).

### 8.2. Statutory NAV Error Tolerance Reference
Any error in the calculation of the NAV per Share is assessed for materiality in strict accordance with the applicable CSSF Circular in force on the relevant Valuation Day:
* For Valuation Days up to and including **31 December 2024**, materiality and investor indemnification thresholds are governed by **Circular CSSF 02/77**.
* For Valuation Days and errors detected as from **1 January 2025**, materiality and compensation rules are governed by **Circular CSSF 24/856**.

---

## Section 9. Share Classes, Management Fees, and Accrual Methodology

### 9.1. Schedule of Share Classes and Annual Management Fees

| Share Class | Target Investor Category | ISIN Code | Base Currency | Annual Management Fee Rate | Subscription Tax (*Taxe d'abonnement*) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Class I (Institutional)** | Institutional Investors | `LU2024000101` | EUR | **0.40% per annum (40 bps)** `(*1)` | 0.01% per annum (1 bps) |
| **Class R (Retail)** | Retail Investors | `LU2024000202` | EUR | **1.20% per annum (120 bps)** | 0.05% per annum (5 bps) |
| **Class H-USD (Hedged)** | USD Currency Hedged | `LU2024000303` | USD | **0.65% per annum (65 bps)** `(*2)` | 0.05% per annum (5 bps) |

**Footnotes to the Fee Schedule (Integral Part of Section 9.1):**
* **Footnote `(*1)` — Tiered Fee Reduction for Class I:** For **Class I (Institutional)** only, a tiered management fee reduction applies based on the sub-fund's Gross Asset Value (GAV): the rate of **0.40% per annum (40 bps)** applies to the first **EUR 50,000,000** of GAV, and a reduced rate of **0.30% per annum (30 bps)** applies to the portion of GAV exceeding **EUR 50,000,000**. No tiered reduction applies to Class R.
* **Footnote `(*2)` — Currency Hedging Cost Allocation:** All gains, losses, and hedging transaction costs of Class H-USD are allocated exclusively to Class H-USD in accordance with CSSF allocation rules (see Circular CSSF 24/856 Paragraph 113).

### 9.2. Day-Count Convention for Fee Accruals
All annual management fees across all Share Classes (Class I, Class R, and Class H-USD) are accrued on each Valuation Day using the **`Actual/365` (Fixed)** day-count convention:

$$\text{Daily Fee Accrual} = \text{Applicable GAV} \times \text{Annual Fee Rate} \times \frac{\text{Actual Calendar Days Elapsed}}{365}$$

> **Strict Compliance Warning:** Using an `Actual/360` or `30/360` denominator for management fee accruals is a breach of Section 9.2 of this Prospectus and systematically overstates fund liabilities by a factor of $365 / 360$ ($+1.3889\%$).
