# Growth-Mode Hypothesis Test (H1) — 2026-05-17

## Overall Verdict: FALSIFIED

H1 is **FALSIFIED**: The growth cohort does not consistently outperform the control cohort. At one or more horizons the growth cohort shows lower mean returns despite adequate cohort sizes.

No horizon meets the 'supported' criteria.

---

## Cohort Definitions

- **Rising ISC universe:** 399 unique ticker × obs_q observations (1,995 rows across 5 horizons)
- **Growth cohort:** rising ISC + Altman `safe` (Z > 2.99) + Piotroski `strong` (F ≥ 7) + Beneish `clean` or `unknown` (ineligible sector treated as pass)
- **Control cohort:** all other rising-ISC observations

---

## Comparison Table

| Horizon | Growth n | Growth mean | Control n | Control mean | Diff mean | Diff 95% CI | CI width | Verdict |
|---------|----------|-------------|-----------|--------------|-----------|-------------|----------|---------|
| 1Q | 37 | -5.66% | 362 | -2.97% | -2.69% | [-8.39%, 3.29%] | 11.68% | falsified |
| 2Q | 37 | -0.49% | 362 | -0.62% | 0.14% | [-7.27%, 7.57%] | 14.84% | inconclusive_signal |
| 4Q | 37 | 13.28% | 362 | 6.84% | 6.44% | [-1.61%, 14.46%] | 16.07% | inconclusive_signal |
| 8Q | 37 | 37.62% | 362 | 47.78% | -10.16% | [-31.13%, 8.07%] | 39.20% | falsified |
| 12Q | 37 | 42.34% | 362 | 62.47% | -20.13% | [-45.30%, 3.56%] | 48.86% | falsified |

---

## Per-Horizon Detail

**1Q** — Growth: mean=-5.66%, median=-7.72%, hit=32.4%, max_dd=-20.21%, CI=[-11.05%, 0.17%]  | Control: mean=-2.97%, median=-4.25%, hit=43.4%, max_dd=-19.39%, CI=[-4.81%, -1.17%]
**2Q** — Growth: mean=-0.49%, median=-1.95%, hit=48.6%, max_dd=-23.49%, CI=[-7.43%, 6.82%]  | Control: mean=-0.62%, median=-1.23%, hit=47.5%, max_dd=-24.03%, CI=[-3.19%, 1.92%]
**4Q** — Growth: mean=13.28%, median=14.38%, hit=67.6%, max_dd=-27.66%, CI=[6.11%, 20.38%]  | Control: mean=6.84%, median=2.94%, hit=54.4%, max_dd=-28.38%, CI=[3.55%, 10.41%]
**8Q** — Growth: mean=37.62%, median=35.98%, hit=83.8%, max_dd=-31.07%, CI=[26.13%, 49.23%]  | Control: mean=47.78%, median=23.61%, hit=71.5%, max_dd=-33.43%, CI=[34.84%, 65.38%]
**12Q** — Growth: mean=42.34%, median=31.37%, hit=78.4%, max_dd=-37.32%, CI=[24.81%, 61.04%]  | Control: mean=62.47%, median=28.24%, hit=71.0%, max_dd=-38.70%, CI=[47.04%, 80.67%]

---

## Sector Breakdown — Growth Cohort (unique obs)

Growth cohort total unique observations: 37

| Sector | Obs count |
|--------|-----------|
| Information Technology | 12 |
| Industrials | 6 |
| Consumer Discretionary | 4 |
| Consumer Staples | 4 |
| Materials | 3 |
| Communication Services | 2 |
| Financials | 2 |
| Energy | 2 |
| Health Care | 1 |
| Real Estate | 1 |

Note: Information Technology accounts for 12/37 (32%) of growth-cohort observations. No single sector dominates.

---

## Pre-Commitment Criteria Applied

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| Supported | Diff mean ≥ CI width at ≥ 2 horizons, both n ≥ 30 | 0 horizon(s) qualify |
| Inconclusive | Positive direction but size < 30 or CI width not cleared | See per-horizon verdict |
| Falsified | Growth underperforms at any horizon with adequate n | See per-horizon verdict |

**Verdict: FALSIFIED**

---

## Notes

- Bootstrap CIs use 10,000 resamples, seed=42.
- `total_return` is cumulative forward total return (not price-only, not excess-of-sector) for the stated horizon.
- `beneish_regime == 'unknown'` rows (Financials/Real Estate ineligible tickers) are treated as passing the Beneish filter.
- No rows were dropped for missing returns (zero nulls across all horizons).
- Raw comparison table saved to: `output\growth_mode_test_2026-05-17.csv`
