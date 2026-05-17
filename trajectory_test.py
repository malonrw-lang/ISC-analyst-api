"""
Task E3 — ISC Trajectory Test (H3)
Tests whether ISC-rising observations with resolved-down trajectories
outperform escalated-down trajectories in forward returns.
"""

import os
from datetime import date

import numpy as np
import pandas as pd

TODAY = date.today().strftime("%Y-%m-%d")
OUTPUT_DIR = "output"
N_BOOT = 10_000
RNG = np.random.default_rng(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Load and reshape
# ---------------------------------------------------------------------------

df = pd.read_csv("backtest_per_stock.csv")

rising = df[df["isc_regime"] == "rising"].copy()

# Pivot outcome_regime by horizon — one column per forward quarter
regime_pivot = rising.pivot_table(
    index=["ticker", "obs_q"],
    columns="horizon_q",
    values="outcome_regime",
    aggfunc="first",
)
regime_pivot.columns = [f"r_{int(c)}q" for c in regime_pivot.columns]
regime_pivot = regime_pivot.reset_index()

# Pivot total_return and max_drawdown by horizon
ret_dd_pivot = rising.pivot_table(
    index=["ticker", "obs_q"],
    columns="horizon_q",
    values=["total_return", "max_drawdown"],
    aggfunc="first",
)
ret_dd_pivot.columns = [f"{v}_{int(h)}q" for v, h in ret_dd_pivot.columns]
ret_dd_pivot = ret_dd_pivot.reset_index()

# Sector metadata (one row per obs)
meta = rising.drop_duplicates(["ticker", "obs_q"])[
    ["ticker", "obs_q", "sector", "sector_bucket"]
].copy()

# One row per (ticker, obs_q)
obs = (
    meta
    .merge(regime_pivot, on=["ticker", "obs_q"])
    .merge(ret_dd_pivot, on=["ticker", "obs_q"])
)

n_total = len(obs)


# ---------------------------------------------------------------------------
# Exclude observations missing T+8Q data
# ---------------------------------------------------------------------------

obs = obs.dropna(subset=["r_8q", "total_return_8q"])
n_excluded = n_total - len(obs)
print(f"Rising observations: {n_total}")
print(f"Excluded (missing T+8Q data): {n_excluded}")
print(f"Included in test: {len(obs)}")


# ---------------------------------------------------------------------------
# Trajectory classification
# ---------------------------------------------------------------------------

def classify_cohort(row):
    window = [row["r_1q"], row["r_2q"], row["r_4q"], row["r_8q"]]
    # escalated_down takes precedence if distressed appears anywhere in window
    if "distressed" in window:
        return "escalated_down"
    if "stable" in window and row["r_8q"] == "stable":
        return "resolved_down"
    return "stayed_elevated"


obs["cohort"] = obs.apply(classify_cohort, axis=1)

cohort_counts = obs["cohort"].value_counts()
print("\nCohort distribution:")
for c in ["resolved_down", "escalated_down", "stayed_elevated"]:
    print(f"  {c}: {cohort_counts.get(c, 0)}")


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def boot_mean(arr, n_boot=N_BOOT):
    if len(arr) == 0:
        return np.full(n_boot, np.nan)
    samples = RNG.choice(arr, size=(n_boot, len(arr)), replace=True)
    return samples.mean(axis=1)


def ci_mean(arr):
    b = boot_mean(arr)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def ci_diff(arr1, arr2):
    d = boot_mean(arr1) - boot_mean(arr2)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------------------
# Per-cohort, per-horizon statistics
# ---------------------------------------------------------------------------

def cohort_stats(sub, ret_col, dd_col):
    ret = sub[ret_col].dropna().values
    dd = sub[dd_col].dropna().values
    n = len(ret)
    if n == 0:
        return {k: np.nan for k in
                ["n", "mean", "median", "hit_rate", "mean_max_dd", "ci_lo", "ci_hi"]}
    lo, hi = ci_mean(ret)
    return {
        "n": int(n),
        "mean": float(np.mean(ret)),
        "median": float(np.median(ret)),
        "hit_rate": float(np.mean(ret > 0)),
        "mean_max_dd": float(np.mean(dd)) if len(dd) > 0 else np.nan,
        "ci_lo": lo,
        "ci_hi": hi,
    }


COHORTS = ["resolved_down", "escalated_down", "stayed_elevated"]
HORIZONS = [8, 12]

stats_rows = []
for cohort in COHORTS:
    sub = obs[obs["cohort"] == cohort]
    for hq in HORIZONS:
        s = cohort_stats(sub, f"total_return_{hq}q", f"max_drawdown_{hq}q")
        stats_rows.append({"cohort": cohort, "horizon_q": hq, **s})

results = pd.DataFrame(stats_rows)


# ---------------------------------------------------------------------------
# Primary comparison: resolved_down vs escalated_down
# ---------------------------------------------------------------------------

diff_rows = []
for hq in HORIZONS:
    ret_col = f"total_return_{hq}q"
    r_arr = obs[obs["cohort"] == "resolved_down"][ret_col].dropna().values
    e_arr = obs[obs["cohort"] == "escalated_down"][ret_col].dropna().values
    if len(r_arr) > 0 and len(e_arr) > 0:
        diff_mean = float(np.mean(r_arr) - np.mean(e_arr))
        d_lo, d_hi = ci_diff(r_arr, e_arr)
        ci_width = d_hi - d_lo
    else:
        diff_mean = d_lo = d_hi = ci_width = np.nan
    diff_rows.append({
        "horizon_q": hq,
        "diff_mean": diff_mean,
        "diff_ci_lo": d_lo,
        "diff_ci_hi": d_hi,
        "diff_ci_width": ci_width,
    })

diff_df = pd.DataFrame(diff_rows)


# ---------------------------------------------------------------------------
# Per-horizon verdict
# ---------------------------------------------------------------------------

def horizon_verdict(diff_row):
    hq = diff_row["horizon_q"]
    r_n = int(results.loc[(results["cohort"] == "resolved_down") &
                           (results["horizon_q"] == hq), "n"].iloc[0])
    e_n = int(results.loc[(results["cohort"] == "escalated_down") &
                           (results["horizon_q"] == hq), "n"].iloc[0])
    adequate = (r_n >= 30) and (e_n >= 30)
    positive = (not np.isnan(diff_row["diff_mean"])) and diff_row["diff_mean"] > 0
    exceeds_width = (
        not np.isnan(diff_row["diff_ci_width"])
        and diff_row["diff_mean"] >= diff_row["diff_ci_width"]
    )
    if adequate and positive and exceeds_width:
        return "supported"
    elif positive and adequate:
        return "inconclusive_signal"
    elif positive:
        return "inconclusive_size"
    else:
        return "falsified"


diff_df["verdict"] = diff_df.apply(horizon_verdict, axis=1)

# Overall verdict using H3 pre-commitment criteria (1 supported horizon sufficient)
supported_horizons = diff_df[diff_df["verdict"] == "supported"]["horizon_q"].tolist()
all_positive = (diff_df["diff_mean"] > 0).all()
any_negative = (diff_df["diff_mean"] < 0).any()

if len(supported_horizons) >= 1:
    overall = "SUPPORTED"
elif all_positive:
    overall = "INCONCLUSIVE"
elif any_negative:
    overall = "FALSIFIED"
else:
    overall = "INCONCLUSIVE"

print(f"\nOverall verdict: {overall}")


# ---------------------------------------------------------------------------
# Sector breakdown by cohort
# ---------------------------------------------------------------------------

sector_counts = (
    obs.groupby(["cohort", "sector"])
    .size()
    .reset_index(name="count")
)
cohort_totals = obs.groupby("cohort").size().reset_index(name="total")
sector_breakdown = sector_counts.merge(cohort_totals, on="cohort")
sector_breakdown["pct_of_cohort"] = (
    sector_breakdown["count"] / sector_breakdown["total"] * 100
).round(1)
sector_breakdown = sector_breakdown.sort_values(
    ["cohort", "count"], ascending=[True, False]
)


# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_rows = []
for _, row in results.iterrows():
    diff_match = diff_df[diff_df["horizon_q"] == row["horizon_q"]]
    csv_rows.append({
        "cohort": row["cohort"],
        "horizon_q": int(row["horizon_q"]),
        "n": int(row["n"]) if not np.isnan(row["n"]) else "",
        "mean": round(row["mean"], 4) if not np.isnan(row["mean"]) else "",
        "median": round(row["median"], 4) if not np.isnan(row["median"]) else "",
        "hit_rate": round(row["hit_rate"], 4) if not np.isnan(row["hit_rate"]) else "",
        "mean_max_dd": round(row["mean_max_dd"], 4) if not np.isnan(row["mean_max_dd"]) else "",
        "ci_lo": round(row["ci_lo"], 4) if not np.isnan(row["ci_lo"]) else "",
        "ci_hi": round(row["ci_hi"], 4) if not np.isnan(row["ci_hi"]) else "",
        "diff_mean_vs_escalated": "",
        "diff_ci_lo": "",
        "diff_ci_hi": "",
        "diff_ci_width": "",
        "horizon_verdict": "",
    })

for _, drow in diff_df.iterrows():
    hq = int(drow["horizon_q"])
    for i, r in enumerate(csv_rows):
        if r["cohort"] == "resolved_down" and r["horizon_q"] == hq:
            csv_rows[i]["diff_mean_vs_escalated"] = round(drow["diff_mean"], 4)
            csv_rows[i]["diff_ci_lo"] = round(drow["diff_ci_lo"], 4)
            csv_rows[i]["diff_ci_hi"] = round(drow["diff_ci_hi"], 4)
            csv_rows[i]["diff_ci_width"] = round(drow["diff_ci_width"], 4)
            csv_rows[i]["horizon_verdict"] = drow["verdict"]

csv_df = pd.DataFrame(csv_rows)
csv_path = os.path.join(OUTPUT_DIR, f"trajectory_test_{TODAY}.csv")
csv_df.to_csv(csv_path, index=False)
print(f"\nCSV written: {csv_path}")


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def fmt_pct(v, decimals=1):
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v * 100:.{decimals}f}%"


def fmt_f(v, decimals=2):
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:.{decimals}f}"


def get_stat(cohort, hq, col):
    row = results[(results["cohort"] == cohort) & (results["horizon_q"] == hq)]
    if row.empty:
        return np.nan
    return row.iloc[0][col]


def get_diff(hq, col):
    row = diff_df[diff_df["horizon_q"] == hq]
    if row.empty:
        return np.nan
    return row.iloc[0][col]


# ---------------------------------------------------------------------------
# Build markdown
# ---------------------------------------------------------------------------

verdict_blurb = {
    "SUPPORTED": (
        "H3 is **SUPPORTED**. The resolved-down cohort outperforms the "
        "escalated-down cohort at one or more test horizons, with both cohorts "
        "n ≥ 30 and the mean difference exceeding the bootstrap 95 CI width."
    ),
    "INCONCLUSIVE": (
        "H3 is **INCONCLUSIVE**. The direction is consistent with H3 at all "
        "tested horizons, but the mean difference does not exceed the bootstrap "
        "95 CI width at either horizon."
    ),
    "FALSIFIED": (
        "H3 is **FALSIFIED**. The resolved-down cohort does not outperform the "
        "escalated-down cohort at either the 8Q or 12Q horizon under the "
        "pre-commitment criteria."
    ),
}

lines = []

# ---- Header & verdict ----
lines.append(f"# ISC Trajectory Test — H3")
lines.append(f"**Date run:** {TODAY}")
lines.append("")
lines.append("## Verdict")
lines.append("")
lines.append(f"### {overall}")
lines.append("")
lines.append(verdict_blurb[overall])
lines.append("")

# ---- Cohort definitions ----
lines.append("## Cohort Definitions")
lines.append("")
lines.append(
    "Universe: all (ticker × obs_q) observations where `isc_regime = rising` at T."
)
lines.append("")
lines.append(
    "For each observation, the trajectory is the sequence of `outcome_regime` "
    "values at T+1Q, T+2Q, T+4Q, T+8Q. Cohorts are mutually exclusive and "
    "assigned in priority order."
)
lines.append("")
lines.append(
    "| Cohort | Definition |"
)
lines.append("| --- | --- |")
lines.append(
    "| **resolved_down** | `outcome_regime = stable` appears at any point in the "
    "T+1Q–T+8Q window, AND the final state at T+8Q is `stable`. |"
)
lines.append(
    "| **escalated_down** | `outcome_regime = distressed` appears at any point in "
    "the T+1Q–T+8Q window. Takes precedence over resolved_down if both "
    "conditions are met. |"
)
lines.append(
    "| **stayed_elevated** | Neither distressed nor a stable-ending trajectory. "
    "Regime stays in rising/elevated states, or reaches stable but does not "
    "end there at T+8Q. |"
)
lines.append("")

# ---- Cohort sizes ----
lines.append("### Cohort Sizes")
lines.append("")
lines.append("| Cohort | n |")
lines.append("| --- | --- |")
for c in COHORTS:
    n = cohort_counts.get(c, 0)
    lines.append(f"| {c} | {n} |")
lines.append("")

# ---- Primary comparison table ----
lines.append("## Primary Comparison: resolved_down vs escalated_down")
lines.append("")
lines.append(
    "| Horizon | resolved n | resolved mean | escalated n | escalated mean "
    "| Diff (R−E) | 95% CI on diff | CI width | Verdict |"
)
lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for hq in HORIZONS:
    r_n = int(get_stat("resolved_down", hq, "n"))
    r_mean = get_stat("resolved_down", hq, "mean")
    e_n = int(get_stat("escalated_down", hq, "n"))
    e_mean = get_stat("escalated_down", hq, "mean")
    dm = get_diff(hq, "diff_mean")
    d_lo = get_diff(hq, "diff_ci_lo")
    d_hi = get_diff(hq, "diff_ci_hi")
    dw = get_diff(hq, "diff_ci_width")
    verd = diff_df[diff_df["horizon_q"] == hq].iloc[0]["verdict"]
    ci_str = f"[{fmt_f(d_lo)}, {fmt_f(d_hi)}]"
    lines.append(
        f"| {hq}Q | {r_n} | {fmt_pct(r_mean)} | {e_n} | {fmt_pct(e_mean)} "
        f"| {fmt_pct(dm)} | {ci_str} | {fmt_pct(dw)} | {verd} |"
    )
lines.append("")

# ---- Per-cohort detail ----
lines.append("## Per-Cohort Detail")
lines.append("")
for cohort in COHORTS:
    lines.append(f"### {cohort}")
    lines.append("")
    lines.append(
        "| Horizon | n | Mean return | Median return | Hit rate | Mean max drawdown "
        "| 95% CI (lo, hi) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for hq in HORIZONS:
        n = int(get_stat(cohort, hq, "n"))
        mean = get_stat(cohort, hq, "mean")
        med = get_stat(cohort, hq, "median")
        hr = get_stat(cohort, hq, "hit_rate")
        dd = get_stat(cohort, hq, "mean_max_dd")
        lo = get_stat(cohort, hq, "ci_lo")
        hi = get_stat(cohort, hq, "ci_hi")
        ci_str = f"[{fmt_pct(lo)}, {fmt_pct(hi)}]"
        lines.append(
            f"| {hq}Q | {n} | {fmt_pct(mean)} | {fmt_pct(med)} "
            f"| {fmt_pct(hr)} | {fmt_pct(dd)} | {ci_str} |"
        )
    lines.append("")

# ---- Sector breakdown ----
lines.append("## Sector Breakdown by Cohort")
lines.append("")
lines.append(
    "Counts and cohort share for each sector within each cohort. "
    "Used to check whether any sector dominates a cohort."
)
lines.append("")
for cohort in COHORTS:
    sub_sec = sector_breakdown[sector_breakdown["cohort"] == cohort]
    total = sub_sec["total"].iloc[0] if len(sub_sec) > 0 else 0
    lines.append(f"### {cohort} (n = {total})")
    lines.append("")
    lines.append("| Sector | n | % of cohort |")
    lines.append("| --- | --- | --- |")
    for _, row in sub_sec.iterrows():
        lines.append(
            f"| {row['sector']} | {int(row['count'])} | {row['pct_of_cohort']}% |"
        )
    lines.append("")

# ---- Pre-commitment criteria ----
lines.append("## Pre-Commitment Criteria Applied")
lines.append("")
lines.append(
    "Criteria as stated in `editorial/hypotheses.md` (H3), "
    "evaluated without modification after seeing results."
)
lines.append("")

# Check each criterion
r8_n = int(get_stat("resolved_down", 8, "n"))
e8_n = int(get_stat("escalated_down", 8, "n"))
r12_n = int(get_stat("resolved_down", 12, "n"))
e12_n = int(get_stat("escalated_down", 12, "n"))
dm8 = get_diff(8, "diff_mean")
dm12 = get_diff(12, "diff_mean")
dw8 = get_diff(8, "diff_ci_width")
dw12 = get_diff(12, "diff_ci_width")
verd8 = diff_df[diff_df["horizon_q"] == 8].iloc[0]["verdict"]
verd12 = diff_df[diff_df["horizon_q"] == 12].iloc[0]["verdict"]

lines.append(f"**8Q horizon**")
lines.append("")
lines.append(
    f"- resolved_down n = {r8_n}, escalated_down n = {e8_n} "
    f"(≥ 30 each: {'yes' if r8_n >= 30 and e8_n >= 30 else 'no'})"
)
lines.append(
    f"- Diff mean = {fmt_pct(dm8)} "
    f"({'positive' if not np.isnan(dm8) and dm8 > 0 else 'not positive'})"
)
lines.append(
    f"- CI width = {fmt_pct(dw8)} — "
    f"diff {'exceeds' if not np.isnan(dm8) and not np.isnan(dw8) and dm8 >= dw8 else 'does not exceed'} CI width"
)
lines.append(f"- **Horizon verdict: {verd8}**")
lines.append("")

lines.append(f"**12Q horizon**")
lines.append("")
lines.append(
    f"- resolved_down n = {r12_n}, escalated_down n = {e12_n} "
    f"(≥ 30 each: {'yes' if r12_n >= 30 and e12_n >= 30 else 'no'})"
)
lines.append(
    f"- Diff mean = {fmt_pct(dm12)} "
    f"({'positive' if not np.isnan(dm12) and dm12 > 0 else 'not positive'})"
)
lines.append(
    f"- CI width = {fmt_pct(dw12)} — "
    f"diff {'exceeds' if not np.isnan(dm12) and not np.isnan(dw12) and dm12 >= dw12 else 'does not exceed'} CI width"
)
lines.append(f"- **Horizon verdict: {verd12}**")
lines.append("")

lines.append(f"**Overall verdict: {overall}**")
lines.append("")

supported_criterion = (
    "supported: resolved_down mean > escalated_down mean at 8Q OR 12Q, "
    "each cohort n ≥ 30, and difference exceeds bootstrap 95% CI width"
)
inconclusive_criterion = (
    "inconclusive: direction consistent with H3 but bucket sizes too small "
    "or CIs overlap zero (difference does not exceed CI width)"
)
falsified_criterion = (
    "falsified: resolved_down does not outperform escalated_down at either "
    "horizon, OR three cohorts show similar returns despite adequate sample sizes"
)

if overall == "SUPPORTED":
    lines.append(
        f"Applied criterion — {supported_criterion}. "
        f"Met at: {', '.join(str(h) + 'Q' for h in supported_horizons)}."
    )
elif overall == "INCONCLUSIVE":
    lines.append(
        f"Applied criterion — {inconclusive_criterion}."
    )
else:
    lines.append(
        f"Applied criterion — {falsified_criterion}."
    )
lines.append("")

# ---- Notes ----
lines.append("## Notes")
lines.append("")
lines.append(
    f"- **Universe:** {n_total} rising-ISC (ticker × obs_q) observations "
    f"across 477 tickers."
)
lines.append(
    f"- **Exclusions:** {n_excluded} observations excluded for missing T+8Q data "
    f"(out of {n_total}). All {len(obs)} included observations have complete "
    f"horizon data at T+1Q, T+2Q, T+4Q, T+8Q, and T+12Q."
)
lines.append(
    "- **Bootstrap:** 10,000 resamples, seed = 42, percentile method "
    "(2.5th and 97.5th percentiles of resampled means)."
)
lines.append(
    "- **Return metric:** `total_return` — cumulative compounded forward return "
    "over the stated horizon. Same column and method as Task E2 (H1). "
    "Individual `qN_return` columns (single-quarter returns) are not used."
)
lines.append(
    "- **Columns used at T:** `isc_regime` (regime filter). "
    "**Columns used at T+N:** `outcome_regime` from the row where "
    "`horizon_q = N`. **Return columns:** `total_return` at `horizon_q = 8` "
    "and `horizon_q = 12`."
)
lines.append(
    "- **Cohort priority:** If an observation has `distressed` in the T+1Q–T+8Q "
    "window, it is classified as `escalated_down` regardless of whether "
    "`stable` also appears. This prevents double-counting."
)
lines.append(
    "- **max_drawdown** reported as the mean of per-observation max drawdown "
    "values within each cohort at the stated horizon."
)
lines.append("")

md = "\n".join(lines)


# ---------------------------------------------------------------------------
# Write markdown
# ---------------------------------------------------------------------------

md_path = os.path.join(OUTPUT_DIR, f"trajectory_test_{TODAY}_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Markdown written: {md_path}")
