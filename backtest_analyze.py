"""
backtest_analyze.py
===================
Read backtest_results.csv and write two reports:

  1. backtest_regime_tables.txt — for each framework, a table showing
     forward stock performance grouped by predicted regime, at each horizon.
     This is the "did distressed stocks actually do worse" view.

  2. backtest_summary.txt — confusion matrices, Spearman rank correlations
     between predicted and realized regimes, and bottom-bucket lift.
     This is the "which framework was most aligned with reality" view.

Run after backtest_pipeline.py finishes.

Usage:
  python backtest_analyze.py [--results backtest_results.csv]

Author: Ryan W. Malone
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Regime ordering (low stress → high stress) per framework
REGIME_ORDER = {
    'isc_regime':       ['stable', 'elevated', 'rising', 'distressed'],
    'altman_regime':    ['safe', 'grey', 'distress'],
    'piotroski_regime': ['strong', 'mixed', 'weak'],
    'beneish_regime':   ['clean', 'manipulator'],
    'composite_regime': ['stable', 'elevated', 'rising', 'distressed'],
    'outcome_regime':   ['stable', 'elevated', 'rising', 'distressed'],
}

FRAMEWORK_LABELS = {
    'isc_regime':       'ISC (Variance EWS)',
    'altman_regime':    'Altman Z',
    'piotroski_regime': 'Piotroski F',
    'beneish_regime':   'Beneish M',
    'composite_regime': 'Composite',
}

HORIZONS = [2, 4, 8, 12]


def regime_rank(regime, framework):
    """Map regime label to integer rank (lower = healthier)."""
    order = REGIME_ORDER.get(framework, [])
    if regime in order:
        return order.index(regime)
    return None


def spearman(x, y):
    """No-scipy Spearman. Returns rho, or NaN if insufficient data."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return float(1 - 6 * np.dot(d, d) / (n * (n * n - 1)))


def fmt_pct(v, places=1):
    if v is None or pd.isna(v):
        return '   N/A'
    return f"{v * 100:+.{places}f}%"


def fmt_count(n):
    return f"{n:>4}"


# ──────────────────────────────────────────────────────────────────────────────
# Report 1: Regime tables
# ──────────────────────────────────────────────────────────────────────────────

def make_regime_tables(df, out_path):
    """
    For each framework, produce a table:
      regime bucket → forward stats per horizon

    Stats per cell: N, mean return, median return, mean max drawdown, hit rate.
    Hit rate = fraction of stocks with positive total return at that horizon.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("FILTER LAB BACKTEST — REGIME PERFORMANCE TABLES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("For each framework, stocks grouped by the framework's predicted regime at Q8.")
    lines.append("Forward stats measured at 2Q, 4Q, 8Q, 12Q after Q8 observation date.")
    lines.append("Healthier-regime buckets should show better returns and smaller drawdowns.")
    lines.append("")

    for framework, label in FRAMEWORK_LABELS.items():
        lines.append("=" * 80)
        lines.append(f"  {label}")
        lines.append("=" * 80)

        order = REGIME_ORDER[framework]
        for h in HORIZONS:
            sub = df[df['horizon_q'] == h]
            if len(sub) == 0:
                continue
            lines.append(f"\n  Horizon: {h}Q forward")
            lines.append(f"  {'Regime':<14} {'N':>5} {'Mean Ret':>10} {'Median Ret':>11} "
                         f"{'Mean DD':>10} {'Mean Vol':>10} {'Hit Rate':>9}")
            lines.append(f"  {'-'*14} {'-'*5} {'-'*10} {'-'*11} {'-'*10} {'-'*10} {'-'*9}")

            for regime in order:
                bucket = sub[sub[framework] == regime]
                n = len(bucket)
                if n == 0:
                    lines.append(f"  {regime:<14} {fmt_count(0)} {'—':>10} {'—':>11} {'—':>10} {'—':>10} {'—':>9}")
                    continue
                mean_ret = bucket['total_return'].mean()
                med_ret = bucket['total_return'].median()
                mean_dd = bucket['max_drawdown'].mean()
                mean_vol = bucket['realized_vol'].mean()
                hit_rate = (bucket['total_return'] > 0).mean()
                lines.append(f"  {regime:<14} {fmt_count(n)} {fmt_pct(mean_ret):>10} {fmt_pct(med_ret):>11} "
                             f"{fmt_pct(mean_dd):>10} {fmt_pct(mean_vol):>10} {hit_rate*100:>7.1f}%")

            # Note unclassified
            n_unknown = (sub[framework] == 'unknown').sum()
            if n_unknown > 0:
                lines.append(f"  {'unknown':<14} {fmt_count(n_unknown)} (regime not computable)")

        lines.append("")

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Report 2: Summary (confusion matrices + rank correlation + lift)
# ──────────────────────────────────────────────────────────────────────────────

def confusion_matrix(df, predicted_col, actual_col='outcome_regime'):
    """
    Build a confusion matrix between predicted_col and outcome_regime.
    Returns DataFrame indexed by predicted, columns are actual.
    """
    pred_order = REGIME_ORDER[predicted_col]
    actual_order = REGIME_ORDER[actual_col]

    sub = df[df[predicted_col].isin(pred_order) & df[actual_col].isin(actual_order)]
    if len(sub) == 0:
        return None

    matrix = pd.crosstab(sub[predicted_col], sub[actual_col])
    # Reindex to canonical order
    matrix = matrix.reindex(index=[r for r in pred_order if r in matrix.index],
                            columns=[r for r in actual_order if r in matrix.columns],
                            fill_value=0)
    return matrix


def fmt_confusion(matrix, framework_label):
    """Pretty-print a confusion matrix."""
    if matrix is None or matrix.empty:
        return f"\n  No data for {framework_label}\n"

    lines = []
    n_total = matrix.values.sum()
    lines.append(f"\n  Confusion matrix: {framework_label} predicted (rows) vs realized outcome (cols)")
    lines.append(f"  Total observations: {n_total}")
    lines.append("")

    col_width = 13
    header = "  " + " " * 16 + "".join(f"{col:>{col_width}}" for col in matrix.columns)
    lines.append(header)
    lines.append("  " + " " * 16 + "".join("-" * col_width for _ in matrix.columns))
    for pred_regime, row in matrix.iterrows():
        cells = "".join(f"{int(v):>{col_width}}" for v in row.values)
        lines.append(f"  {pred_regime:<16}{cells}")

    # Row totals
    row_totals = matrix.sum(axis=1)
    lines.append("")
    lines.append("  Row %s (predicted regime distribution):")
    for pred, total in row_totals.items():
        pct = 100.0 * total / n_total if n_total > 0 else 0
        lines.append(f"    {pred:<14} {int(total):>5} ({pct:.1f}%)")

    return "\n".join(lines)


def compute_lift(df, predicted_col):
    """
    Lift in the worst predicted bucket: of stocks predicted in the worst regime
    by this framework, what fraction actually realized as 'distressed'?
    Compare to base rate.
    """
    pred_order = REGIME_ORDER[predicted_col]
    if not pred_order:
        return None
    worst_pred = pred_order[-1]

    sub = df[df[predicted_col].isin(pred_order)]
    if len(sub) == 0:
        return None

    base_rate = (sub['outcome_regime'] == 'distressed').mean()
    bottom = sub[sub[predicted_col] == worst_pred]
    if len(bottom) == 0:
        return {'worst_pred': worst_pred, 'n_bottom': 0, 'base_rate': base_rate,
                'bottom_rate': None, 'lift': None}
    bottom_rate = (bottom['outcome_regime'] == 'distressed').mean()
    lift = bottom_rate / base_rate if base_rate > 0 else None
    return {
        'worst_pred': worst_pred,
        'n_bottom': len(bottom),
        'base_rate': base_rate,
        'bottom_rate': bottom_rate,
        'lift': lift,
    }


def make_summary(df, out_path):
    lines = []
    lines.append("=" * 80)
    lines.append("FILTER LAB BACKTEST — SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total observations (ticker × horizon): {len(df)}")
    lines.append(f"Unique tickers: {df['ticker'].nunique()}")
    lines.append(f"Date range of Q8 observations: {df['obs_date'].min()} to {df['obs_date'].max()}")
    lines.append("")

    # ── Section 1: Sample composition ────────────────────────────────────────
    lines.append("=" * 80)
    lines.append("  SAMPLE COMPOSITION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Sector bucket distribution (tickers, not observations):")
    unique_tickers = df.drop_duplicates('ticker')
    for sec, n in unique_tickers['sector_bucket'].value_counts().items():
        lines.append(f"    {sec:<14} {n}")
    lines.append("")
    lines.append("  Realized outcome regime distribution at each horizon:")
    for h in HORIZONS:
        sub = df[df['horizon_q'] == h]
        if len(sub) == 0:
            continue
        lines.append(f"\n    {h}Q horizon (N={len(sub)}):")
        for outcome in REGIME_ORDER['outcome_regime']:
            n = (sub['outcome_regime'] == outcome).sum()
            pct = 100.0 * n / len(sub) if len(sub) > 0 else 0
            lines.append(f"      {outcome:<14} {n:>4} ({pct:.1f}%)")

    lines.append("")

    # ── Section 2: Rank correlation (the headline number) ────────────────────
    lines.append("=" * 80)
    lines.append("  HEADLINE: SPEARMAN RANK CORRELATION (predicted regime vs realized)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Each framework's regime rank ordering vs the realized outcome's.")
    lines.append("  Positive rho = framework's worse buckets correspond to worse realized outcomes.")
    lines.append("  Negative rho = inverse relationship (predicting opposite of what happens).")
    lines.append("  Near zero = no relationship.")
    lines.append("")
    lines.append(f"  {'Framework':<20} {'2Q rho':>10} {'4Q rho':>10} {'8Q rho':>10} {'12Q rho':>10}")
    lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    headline_results = []
    for framework, label in FRAMEWORK_LABELS.items():
        cells = [f"  {label:<20}"]
        for h in HORIZONS:
            sub = df[df['horizon_q'] == h].copy()
            sub['pred_rank'] = sub[framework].apply(lambda r: regime_rank(r, framework))
            sub['actual_rank'] = sub['outcome_regime'].apply(lambda r: regime_rank(r, 'outcome_regime'))
            valid = sub.dropna(subset=['pred_rank', 'actual_rank'])
            if len(valid) < 10:
                cells.append(f"{'N/A':>10}")
                continue
            rho = spearman(valid['pred_rank'].values, valid['actual_rank'].values)
            cells.append(f"{rho:>10.3f}")
            headline_results.append((framework, label, h, rho, len(valid)))
        lines.append("".join(cells))

    # Find best framework at each horizon
    lines.append("")
    lines.append("  Best framework per horizon (highest Spearman rho):")
    for h in HORIZONS:
        h_results = [r for r in headline_results if r[2] == h]
        if not h_results:
            continue
        best = max(h_results, key=lambda r: r[3])
        lines.append(f"    {h}Q: {best[1]} (rho = {best[3]:.3f}, N = {best[4]})")
    lines.append("")

    # ── Section 3: Lift in worst-bucket prediction ───────────────────────────
    lines.append("=" * 80)
    lines.append("  LIFT IN WORST-PREDICTED-BUCKET")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Of stocks in the framework's worst-predicted regime, what fraction")
    lines.append("  realized as 'distressed' (return < -25% OR max drawdown < -50%)?")
    lines.append("  Lift = (bottom-bucket rate) / (base rate). >1.0 means useful signal.")
    lines.append("")
    lines.append(f"  {'Framework':<20} {'Horizon':<8} {'Bottom Pred':<14} {'N':>5} {'Base':>7} {'Bottom':>8} {'Lift':>7}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*14} {'-'*5} {'-'*7} {'-'*8} {'-'*7}")
    for framework, label in FRAMEWORK_LABELS.items():
        for h in HORIZONS:
            sub = df[df['horizon_q'] == h]
            if len(sub) == 0:
                continue
            r = compute_lift(sub, framework)
            if r is None:
                continue
            base = f"{r['base_rate']*100:>5.1f}%" if r['base_rate'] is not None else 'N/A'
            bot = f"{r['bottom_rate']*100:>6.1f}%" if r['bottom_rate'] is not None else 'N/A'
            lift = f"{r['lift']:>5.2f}x" if r['lift'] is not None else 'N/A'
            lines.append(f"  {label:<20} {h}Q       {r['worst_pred']:<14} "
                         f"{r['n_bottom']:>5} {base:>7} {bot:>8} {lift:>7}")
    lines.append("")

    # ── Section 4: Confusion matrices (4Q and 8Q, the most interpretable) ────
    lines.append("=" * 80)
    lines.append("  CONFUSION MATRICES (4Q and 8Q horizons)")
    lines.append("=" * 80)
    for h in [4, 8]:
        sub = df[df['horizon_q'] == h]
        lines.append(f"\n  ───── {h}Q forward ─────")
        for framework, label in FRAMEWORK_LABELS.items():
            matrix = confusion_matrix(sub, framework)
            lines.append(fmt_confusion(matrix, label))

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='backtest_results.csv')
    parser.add_argument('--tables-out', default='backtest_regime_tables.txt')
    parser.add_argument('--summary-out', default='backtest_summary.txt')
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    print(f"  Horizons present: {sorted(df['horizon_q'].unique())}")
    print()

    make_regime_tables(df, args.tables_out)
    make_summary(df, args.summary_out)


if __name__ == '__main__':
    main()
