# ISC Trajectory Test — H3
**Date run:** 2026-05-17

## Verdict

### INCONCLUSIVE

H3 is **INCONCLUSIVE**. The direction is consistent with H3 at all tested horizons, but the mean difference does not exceed the bootstrap 95 CI width at either horizon.

## Methodological Caveats

1. **Sample concentration:** 7 of 10 resolved_down observations share obs_date 2022-09-30, near the 2022 bear-market low. The cohort's strong forward returns therefore conflate trajectory effects with entry-point timing. A trajectory-only effect would require a sample distributed across multiple entry points.

2. **Outlier influence:** IRM at +190.3% (8Q) is a single observation that contributes disproportionately to the resolved_down cohort mean. Median return (+82.4%) is more representative than mean (+83.9%).

3. **Escalated_down is more robust:** n=113 spread across the full backtest window. The finding for escalated_down (median return -12.2% at 8Q, 40.7% hit rate, mean max drawdown -48.2%) does not share the timing concentration issue.

4. **Stayed_elevated as baseline:** this cohort represents 69% of rising-ISC observations (n=276) and shows strong typical returns (median +34.6% at 8Q, hit rate 84.8%). Any narrative about rising ISC meaning trouble has to contend with the fact that most rising-ISC observations do fine.

## Cohort Definitions

Universe: all (ticker × obs_q) observations where `isc_regime = rising` at T.

For each observation, the trajectory is the sequence of `outcome_regime` values at T+1Q, T+2Q, T+4Q, T+8Q. Cohorts are mutually exclusive and assigned in priority order.

| Cohort | Definition |
| --- | --- |
| **resolved_down** | `outcome_regime = stable` appears at any point in the T+1Q–T+8Q window, AND the final state at T+8Q is `stable`. |
| **escalated_down** | `outcome_regime = distressed` appears at any point in the T+1Q–T+8Q window. Takes precedence over resolved_down if both conditions are met. |
| **stayed_elevated** | Neither distressed nor a stable-ending trajectory. Regime stays in rising/elevated states, or reaches stable but does not end there at T+8Q. |

### Cohort Sizes

| Cohort | n |
| --- | --- |
| resolved_down | 10 |
| escalated_down | 113 |
| stayed_elevated | 276 |

## Primary Comparison: resolved_down vs escalated_down

| Horizon | resolved n | resolved mean | escalated n | escalated mean | Diff (R-E) | 95% CI on diff | CI width | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8Q | 10 | 83.9% | 113 | 8.7% | 75.2% | [0.43, 1.09] | 65.5% | inconclusive_size |
| 12Q | 10 | 91.4% | 113 | 20.1% | 71.3% | [0.23, 1.12] | 88.6% | inconclusive_size |

## Per-Cohort Detail

### resolved_down

| Horizon | n | Mean return | Median return | Hit rate | Mean max drawdown | 95% CI (lo, hi) |
| --- | --- | --- | --- | --- | --- | --- |
| 8Q | 10 | 83.9% | 82.4% | 100.0% | -12.5% | [57.2%, 114.5%] |
| 12Q | 10 | 91.4% | 95.6% | 100.0% | -21.8% | [61.7%, 119.9%] |

### escalated_down

| Horizon | n | Mean return | Median return | Hit rate | Mean max drawdown | 95% CI (lo, hi) |
| --- | --- | --- | --- | --- | --- | --- |
| 8Q | 113 | 8.7% | -12.2% | 40.7% | -48.2% | [-5.6%, 27.1%] |
| 12Q | 113 | 20.1% | -14.4% | 40.7% | -51.0% | [-5.1%, 61.1%] |

### stayed_elevated

| Horizon | n | Mean return | Median return | Hit rate | Mean max drawdown | 95% CI (lo, hi) |
| --- | --- | --- | --- | --- | --- | --- |
| 8Q | 276 | 61.1% | 34.6% | 84.8% | -27.8% | [46.0%, 82.7%] |
| 12Q | 276 | 76.1% | 39.7% | 83.3% | -34.1% | [59.9%, 94.5%] |

## Sector Breakdown by Cohort

Counts and cohort share for each sector within each cohort. Used to check whether any sector dominates a cohort.

### resolved_down (n = 10)

| Sector | n | % of cohort |
| --- | --- | --- |
| Consumer Discretionary | 2 | 20.0% |
| Consumer Staples | 2 | 20.0% |
| Materials | 2 | 20.0% |
| Financials | 1 | 10.0% |
| Industrials | 1 | 10.0% |
| Real Estate | 1 | 10.0% |
| Utilities | 1 | 10.0% |

### escalated_down (n = 113)

| Sector | n | % of cohort |
| --- | --- | --- |
| Information Technology | 17 | 15.0% |
| Health Care | 16 | 14.2% |
| Consumer Discretionary | 15 | 13.3% |
| Financials | 15 | 13.3% |
| Communication Services | 11 | 9.7% |
| Industrials | 10 | 8.8% |
| Real Estate | 9 | 8.0% |
| Materials | 8 | 7.1% |
| Consumer Staples | 6 | 5.3% |
| Utilities | 4 | 3.5% |
| Energy | 2 | 1.8% |

### stayed_elevated (n = 276)

| Sector | n | % of cohort |
| --- | --- | --- |
| Information Technology | 56 | 20.3% |
| Industrials | 46 | 16.7% |
| Financials | 35 | 12.7% |
| Health Care | 31 | 11.2% |
| Consumer Discretionary | 28 | 10.1% |
| Materials | 18 | 6.5% |
| Utilities | 15 | 5.4% |
| Energy | 14 | 5.1% |
| Consumer Staples | 13 | 4.7% |
| Real Estate | 12 | 4.3% |
| Communication Services | 8 | 2.9% |

## Pre-Commitment Criteria Applied

Criteria as stated in `editorial/hypotheses.md` (H3), evaluated without modification after seeing results.

**8Q horizon**

- resolved_down n = 10, escalated_down n = 113 (≥ 30 each: no)
- Diff mean = 75.2% (positive)
- CI width = 65.5% — diff exceeds CI width
- **Horizon verdict: inconclusive_size**

**12Q horizon**

- resolved_down n = 10, escalated_down n = 113 (≥ 30 each: no)
- Diff mean = 71.3% (positive)
- CI width = 88.6% — diff does not exceed CI width
- **Horizon verdict: inconclusive_size**

**Overall verdict: INCONCLUSIVE**

Applied criterion — inconclusive: direction consistent with H3 but bucket sizes too small or CIs overlap zero (difference does not exceed CI width).

## Notes

- **Universe:** 399 rising-ISC (ticker × obs_q) observations across 477 tickers.
- **Exclusions:** 0 observations excluded for missing T+8Q data (out of 399). All 399 included observations have complete horizon data at T+1Q, T+2Q, T+4Q, T+8Q, and T+12Q.
- **Bootstrap:** 10,000 resamples, seed = 42, percentile method (2.5th and 97.5th percentiles of resampled means).
- **Return metric:** `total_return` — cumulative compounded forward return over the stated horizon. Same column and method as Task E2 (H1). Individual `qN_return` columns (single-quarter returns) are not used.
- **Columns used at T:** `isc_regime` (regime filter). **Columns used at T+N:** `outcome_regime` from the row where `horizon_q = N`. **Return columns:** `total_return` at `horizon_q = 8` and `horizon_q = 12`.
- **Cohort priority:** If an observation has `distressed` in the T+1Q–T+8Q window, it is classified as `escalated_down` regardless of whether `stable` also appears. This prevents double-counting.
- **max_drawdown** reported as the mean of per-observation max drawdown values within each cohort at the stated horizon.
