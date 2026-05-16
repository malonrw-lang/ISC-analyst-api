# Conditional Hit-Rate Aggregator
#
# Assumptions:
#   - Regime risk ordering (best → worst): stable < rising < elevated < distressed
#   - "Reversion" means outcome moves to a LOWER-risk regime than observed (toward stable)
#   - "Deterioration" means outcome moves to a HIGHER-risk regime than observed (toward distressed)
#   - "Persistence" means outcome_regime == composite_regime
#   - Each row in the CSV is already one (ticker, obs_q, horizon_q) triple — no pivoting needed
#   - composite_regime is the observed regime; outcome_regime is the realized regime at horizon_q
#   - Rows with missing composite_regime or outcome_regime are excluded per horizon bucket and logged
#   - Any regime label not in the known set causes a hard failure listing the unknown values
#   - Buckets with zero rows are silently skipped (not emitted)
#   - Output directory is created if it does not exist

import argparse
import logging
import os
import sys
from datetime import date

import pandas as pd

KNOWN_REGIMES = ["stable", "rising", "elevated", "distressed"]
REGIME_RANK = {r: i for i, r in enumerate(KNOWN_REGIMES)}  # stable=0 … distressed=3

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate backtest CSV into conditional hit-rate table.")
    parser.add_argument(
        "--input",
        default="backtest_per_stock.csv",
        help="Path to backtest_per_stock.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--horizons",
        default="1,2,4,8,12",
        help="Comma-separated horizon_q values to include (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: output/conditional_hit_rates_<date>.csv)",
    )
    return parser.parse_args()


def validate_regimes(df: pd.DataFrame) -> None:
    for col in ("composite_regime", "outcome_regime"):
        unknown = set(df[col].dropna().unique()) - set(KNOWN_REGIMES)
        if unknown:
            sys.exit(f"ERROR: Unknown regime labels in '{col}': {sorted(unknown)}")


def compute_bucket(group: pd.DataFrame) -> dict:
    n = len(group)
    counts = group["outcome_regime"].value_counts()

    pcts = {f"pct_{r}": round(counts.get(r, 0) / n * 100, 2) for r in KNOWN_REGIMES}

    modal = counts.idxmax() if n > 0 else None
    observed_rank = REGIME_RANK[group["composite_regime"].iloc[0]]

    persistence = sum(REGIME_RANK[r] == observed_rank for r in group["outcome_regime"])
    reversion = sum(REGIME_RANK[r] < observed_rank for r in group["outcome_regime"])
    deterioration = sum(REGIME_RANK[r] > observed_rank for r in group["outcome_regime"])

    return {
        "n": n,
        "modal_outcome": modal,
        **pcts,
        "persistence_rate": round(persistence / n * 100, 2),
        "reversion_rate": round(reversion / n * 100, 2),
        "deterioration_rate": round(deterioration / n * 100, 2),
        "low_confidence_flag": n < 10,
    }


def main():
    args = parse_args()
    horizons = [int(h.strip()) for h in args.horizons.split(",")]
    out_path = args.out or os.path.join("output", f"conditional_hit_rates_{date.today().isoformat()}.csv")

    # Load
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows from %s", len(df), args.input)

    # Validate regime labels before doing anything else
    validate_regimes(df)

    # Filter to requested horizons
    df = df[df["horizon_q"].isin(horizons)].copy()
    log.info("Rows after horizon filter (%s): %d", args.horizons, len(df))

    # Track exclusions
    missing_mask = df["composite_regime"].isna() | df["outcome_regime"].isna()
    n_missing = missing_mask.sum()
    if n_missing:
        log.warning("Excluding %d rows with missing composite_regime or outcome_regime", n_missing)
    df = df[~missing_mask]

    # Build output rows
    records = []
    for (obs_regime, horizon), group in df.groupby(["composite_regime", "horizon_q"], sort=True):
        if len(group) == 0:
            continue
        row = {"observed_regime": obs_regime, "horizon_q": horizon}
        row.update(compute_bucket(group))
        records.append(row)

    out_df = pd.DataFrame(records, columns=[
        "observed_regime", "horizon_q", "n", "modal_outcome",
        "pct_stable", "pct_rising", "pct_elevated", "pct_distressed",
        "persistence_rate", "reversion_rate", "deterioration_rate", "low_confidence_flag",
    ])

    # Reconciliation: total n at horizon_q=1 should ≈ input row count (minus exclusions)
    h1_total = out_df[out_df["horizon_q"] == 1]["n"].sum()
    input_h1_rows = len(df[df["horizon_q"] == 1])
    print(f"\n--- Reconciliation (horizon_q=1) ---")
    print(f"  Input rows at horizon_q=1 : {input_h1_rows}")
    print(f"  Sum of n across buckets   : {h1_total}")
    print(f"  Excluded (missing regime) : {n_missing}")
    print(f"  Match: {'YES' if h1_total == input_h1_rows else 'NO — check exclusion logic'}")

    # Top 5 most populous buckets (by n, at any horizon)
    top5 = out_df.nlargest(5, "n")[
        ["observed_regime", "horizon_q", "n", "modal_outcome", "persistence_rate"]
    ]
    print(f"\n--- Top 5 most populous buckets ---")
    print(top5.to_string(index=False))

    # Write output
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("Wrote %d rows to %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
