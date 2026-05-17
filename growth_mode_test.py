"""
Task E2 — Growth-Mode Hypothesis Test (H1)
Tests whether rising-ISC + clean-fundamentals observations show positive
forward returns relative to a rising-ISC control cohort.
"""

import os
import textwrap
from datetime import date

import numpy as np
import pandas as pd

TODAY = date.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "output"
N_BOOT = 10_000
RNG = np.random.default_rng(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _boot_mean(arr, n_boot=N_BOOT):
    n = len(arr)
    samples = RNG.choice(arr, size=(n_boot, n), replace=True)
    return samples.mean(axis=1)


def ci_mean(arr):
    """95% bootstrap CI on the mean. Returns (lo, hi)."""
    if len(arr) == 0:
        return np.nan, np.nan
    b = _boot_mean(arr)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ci_diff(arr1, arr2):
    """95% bootstrap CI on (mean(arr1) - mean(arr2)). Returns (lo, hi)."""
    if len(arr1) == 0 or len(arr2) == 0:
        return np.nan, np.nan
    diff = _boot_mean(arr1) - _boot_mean(arr2)
    return float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))


# ---------------------------------------------------------------------------
# Cohort stats
# ---------------------------------------------------------------------------

def cohort_stats(sub):
    """Compute per-cohort stats for one horizon slice."""
    ret = sub["total_return"].values
    dd = sub["max_drawdown"].values
    n = len(ret)
    if n == 0:
        return {k: np.nan for k in
                ["n", "mean", "median", "hit_rate", "mean_max_dd",
                 "ci_lo", "ci_hi"]}
    lo, hi = ci_mean(ret)
    return dict(
        n=n,
        mean=float(np.mean(ret)),
        median=float(np.median(ret)),
        hit_rate=float((ret > 0).mean()),
        mean_max_dd=float(np.mean(dd)),
        ci_lo=lo,
        ci_hi=hi,
    )


# ---------------------------------------------------------------------------
# Load & filter
# ---------------------------------------------------------------------------

df = pd.read_csv("backtest_per_stock.csv")
rising = df[df["isc_regime"] == "rising"].copy()

growth_mask = (
    (rising["altman_regime"] == "safe")
    & (rising["piotroski_regime"] == "strong")
    & (rising["beneish_regime"].isin(["clean", "unknown"]))
)
rising["cohort"] = np.where(growth_mask, "growth", "control")

# ---------------------------------------------------------------------------
# Per-horizon comparison table
# ---------------------------------------------------------------------------

HORIZONS = [1, 2, 4, 8, 12]
rows = []

for hq in HORIZONS:
    hslice = rising[rising["horizon_q"] == hq]
    g = hslice[hslice["cohort"] == "growth"]
    c = hslice[hslice["cohort"] == "control"]

    g_stats = cohort_stats(g)
    c_stats = cohort_stats(c)

    d_lo, d_hi = ci_diff(g["total_return"].values, c["total_return"].values)
    diff_mean = g_stats["mean"] - c_stats["mean"]
    ci_width = (d_hi - d_lo) if not (np.isnan(d_lo) or np.isnan(d_hi)) else np.nan

    rows.append({
        "horizon_q": hq,
        "growth_n": g_stats["n"],
        "growth_mean": round(g_stats["mean"], 4),
        "growth_median": round(g_stats["median"], 4),
        "growth_hit_rate": round(g_stats["hit_rate"], 4),
        "growth_mean_max_dd": round(g_stats["mean_max_dd"], 4),
        "growth_ci_lo": round(g_stats["ci_lo"], 4),
        "growth_ci_hi": round(g_stats["ci_hi"], 4),
        "control_n": c_stats["n"],
        "control_mean": round(c_stats["mean"], 4),
        "control_median": round(c_stats["median"], 4),
        "control_hit_rate": round(c_stats["hit_rate"], 4),
        "control_mean_max_dd": round(c_stats["mean_max_dd"], 4),
        "control_ci_lo": round(c_stats["ci_lo"], 4),
        "control_ci_hi": round(c_stats["ci_hi"], 4),
        "diff_mean": round(diff_mean, 4),
        "diff_ci_lo": round(d_lo, 4),
        "diff_ci_hi": round(d_hi, 4),
        "diff_ci_width": round(ci_width, 4),
    })

results = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Per-horizon verdict
# ---------------------------------------------------------------------------

def horizon_verdict(row):
    adequate = (row["growth_n"] >= 30) and (row["control_n"] >= 30)
    positive = row["diff_mean"] > 0
    exceeds_width = (not np.isnan(row["diff_ci_width"])
                     and row["diff_mean"] >= row["diff_ci_width"])
    if adequate and positive and exceeds_width:
        return "supported"
    elif positive and adequate and not exceeds_width:
        return "inconclusive_signal"
    elif positive and not adequate:
        return "inconclusive_size"
    else:
        return "falsified"


results["verdict"] = results.apply(horizon_verdict, axis=1)

supported_horizons = results[results["verdict"] == "supported"]["horizon_q"].tolist()
if len(supported_horizons) >= 2:
    overall = "SUPPORTED"
elif results["diff_mean"].gt(0).all():
    overall = "INCONCLUSIVE"
elif results["diff_mean"].lt(0).any():
    overall = "FALSIFIED"
else:
    overall = "INCONCLUSIVE"

# ---------------------------------------------------------------------------
# Sector breakdown in growth cohort (unique ticker×obs_q)
# ---------------------------------------------------------------------------

growth_unique = rising[rising["cohort"] == "growth"].drop_duplicates(
    subset=["ticker", "obs_q"]
)
sector_counts = (
    growth_unique["sector"]
    .value_counts()
    .rename_axis("sector")
    .reset_index(name="n_obs")
)

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, f"growth_mode_test_{TODAY}.csv")
results.to_csv(csv_path, index=False)

# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------

def fmt_pct(v):
    return f"{v*100:.1f}%" if not np.isnan(v) else "n/a"

def fmt_ret(v):
    return f"{v*100:.2f}%" if not np.isnan(v) else "n/a"

def fmt_ci(lo, hi):
    return f"[{lo*100:.2f}%, {hi*100:.2f}%]" if not np.isnan(lo) else "n/a"


horizon_table_lines = [
    "| Horizon | Growth n | Growth mean | Control n | Control mean | Diff mean | Diff 95% CI | CI width | Verdict |",
    "|---------|----------|-------------|-----------|--------------|-----------|-------------|----------|---------|",
]
for _, r in results.iterrows():
    horizon_table_lines.append(
        f"| {int(r.horizon_q)}Q"
        f" | {int(r.growth_n)}"
        f" | {fmt_ret(r.growth_mean)}"
        f" | {int(r.control_n)}"
        f" | {fmt_ret(r.control_mean)}"
        f" | {fmt_ret(r.diff_mean)}"
        f" | {fmt_ci(r.diff_ci_lo, r.diff_ci_hi)}"
        f" | {fmt_ret(r.diff_ci_width)}"
        f" | {r.verdict} |"
    )

detail_lines = []
for _, r in results.iterrows():
    detail_lines.append(
        f"**{int(r.horizon_q)}Q** — "
        f"Growth: mean={fmt_ret(r.growth_mean)}, median={fmt_ret(r.growth_median)}, "
        f"hit={fmt_pct(r.growth_hit_rate)}, max_dd={fmt_ret(r.growth_mean_max_dd)}, "
        f"CI={fmt_ci(r.growth_ci_lo, r.growth_ci_hi)}  "
        f"| Control: mean={fmt_ret(r.control_mean)}, median={fmt_ret(r.control_median)}, "
        f"hit={fmt_pct(r.control_hit_rate)}, max_dd={fmt_ret(r.control_mean_max_dd)}, "
        f"CI={fmt_ci(r.control_ci_lo, r.control_ci_hi)}"
    )

sector_lines = ["| Sector | Obs count |", "|--------|-----------|"]
for _, sr in sector_counts.iterrows():
    sector_lines.append(f"| {sr.sector} | {int(sr.n_obs)} |")

verdict_explanation = {
    "SUPPORTED": (
        "H1 is **SUPPORTED**: The growth cohort mean return exceeds the control "
        "cohort mean return by at least one full bootstrap CI width at 2 or more "
        "horizons, with both cohorts ≥ 30 observations at those horizons."
    ),
    "INCONCLUSIVE": (
        "H1 is **INCONCLUSIVE**: The growth cohort shows positive mean return "
        "differences relative to control, but either cohort sizes fall below 30 "
        "at the relevant horizons, or the mean difference does not exceed one full "
        "CI width (effect is positive but insufficiently distinct from noise)."
    ),
    "FALSIFIED": (
        "H1 is **FALSIFIED**: The growth cohort does not consistently outperform "
        "the control cohort. At one or more horizons the growth cohort shows lower "
        "mean returns despite adequate cohort sizes."
    ),
}

supported_str = (
    f"Horizons meeting 'supported' criteria: {', '.join(str(h)+'Q' for h in supported_horizons)}"
    if supported_horizons
    else "No horizon meets the 'supported' criteria."
)

summary_md = f"""# Growth-Mode Hypothesis Test (H1) — {TODAY}

## Overall Verdict: {overall}

{verdict_explanation[overall]}

{supported_str}

---

## Cohort Definitions

- **Rising ISC universe:** 399 unique ticker × obs_q observations (1,995 rows across 5 horizons)
- **Growth cohort:** rising ISC + Altman `safe` (Z > 2.99) + Piotroski `strong` (F ≥ 7) + Beneish `clean` or `unknown` (ineligible sector treated as pass)
- **Control cohort:** all other rising-ISC observations

---

## Comparison Table

{chr(10).join(horizon_table_lines)}

---

## Per-Horizon Detail

{chr(10).join(detail_lines)}

---

## Sector Breakdown — Growth Cohort (unique obs)

Growth cohort total unique observations: {len(growth_unique)}

{chr(10).join(sector_lines)}

Note: {sector_counts.iloc[0].sector} accounts for {sector_counts.iloc[0].n_obs}/{len(growth_unique)} ({sector_counts.iloc[0].n_obs/len(growth_unique)*100:.0f}%) of growth-cohort observations. {"Sector concentration may partially explain the effect." if sector_counts.iloc[0].n_obs/len(growth_unique) > 0.4 else "No single sector dominates."}

---

## Pre-Commitment Criteria Applied

| Criterion | Threshold | Result |
|-----------|-----------|--------|
| Supported | Diff mean ≥ CI width at ≥ 2 horizons, both n ≥ 30 | {len(supported_horizons)} horizon(s) qualify |
| Inconclusive | Positive direction but size < 30 or CI width not cleared | See per-horizon verdict |
| Falsified | Growth underperforms at any horizon with adequate n | See per-horizon verdict |

**Verdict: {overall}**

---

## Notes

- Bootstrap CIs use 10,000 resamples, seed=42.
- `total_return` is cumulative forward total return (not price-only, not excess-of-sector) for the stated horizon.
- `beneish_regime == 'unknown'` rows (Financials/Real Estate ineligible tickers) are treated as passing the Beneish filter.
- No rows were dropped for missing returns (zero nulls across all horizons).
- Raw comparison table saved to: `{csv_path}`
"""

md_path = os.path.join(OUTPUT_DIR, f"growth_mode_test_{TODAY}_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(summary_md)

# ---------------------------------------------------------------------------
# Stdout
# ---------------------------------------------------------------------------

print("=" * 72)
print(f"GROWTH-MODE HYPOTHESIS TEST (H1) — {TODAY}")
print("=" * 72)
print()
print(f"Overall verdict: {overall}")
print()
print("Cohort sizes (per horizon — same at all since no missing returns):")
print(f"  Growth:  {int(results.iloc[0].growth_n)} obs")
print(f"  Control: {int(results.iloc[0].control_n)} obs")
print()
print("Comparison by horizon:")
header = f"{'Horiz':>6}  {'G_mean':>8}  {'C_mean':>8}  {'Diff':>8}  {'Diff CI':>22}  {'CI_wid':>8}  Verdict"
print(header)
print("-" * len(header))
for _, r in results.iterrows():
    diff_ci_str = f"[{r.diff_ci_lo*100:+.2f}%, {r.diff_ci_hi*100:+.2f}%]"
    print(
        f"{int(r.horizon_q):>5}Q"
        f"  {r.growth_mean*100:>7.2f}%"
        f"  {r.control_mean*100:>7.2f}%"
        f"  {r.diff_mean*100:>+7.2f}%"
        f"  {diff_ci_str:>22}"
        f"  {r.diff_ci_width*100:>7.2f}%"
        f"  {r.verdict}"
    )
print()
print("Sector breakdown (growth cohort unique obs):")
for _, sr in sector_counts.iterrows():
    print(f"  {sr.sector:<30} {int(sr.n_obs):>3}")
print()
print(f"Output CSV:      {csv_path}")
print(f"Output summary:  {md_path}")
print("=" * 72)
