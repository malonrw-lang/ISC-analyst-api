"""
backtest_analyze.py — v2
========================
Read backtest_results.csv and produce four reports:

  1. backtest_regime_tables.txt   — regime-bucket performance tables (v1, unchanged)
  2. backtest_summary.txt         — confusion matrices, Spearman, lift (v1, unchanged)
  3. backtest_per_stock.csv       — one row per ticker with all four regimes,
                                    quarter-by-quarter returns, sector-relative
                                    returns at each horizon, and which framework
                                    was 'closest-aligned' with the realized outcome.
  4. backtest_narratives.txt      — human-readable per-stock paragraphs using
                                    'closest-aligned' framing (not 'predicted').

CRITICAL FRAMING NOTE:
  We use 'closest-aligned with realized outcome' rather than 'predicted' for
  per-stock attribution. With Spearman rho ~0.26 (ISC at best horizon), no
  single framework 'predicted' anything for a specific stock; we can only say
  which framework's regime ordering was most consistent with how the stock
  actually traded post-observation.

Usage:
  python backtest_analyze.py [--results backtest_results.csv]

Author: Ryan W. Malone
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REGIME_ORDER = {
    'isc_regime':       ['stable', 'elevated', 'rising', 'distressed'],
    'altman_regime':    ['safe', 'grey', 'distress'],
    'piotroski_regime': ['strong', 'mixed', 'weak'],
    'beneish_regime':   ['clean', 'manipulator'],
    'composite_regime': ['stable', 'elevated', 'rising', 'distressed'],
    'outcome_regime':   ['stable', 'elevated', 'rising', 'distressed'],
}

# Stress score per framework (0 = healthiest, higher = more stressed).
# Normalized to a 0-3 scale so frameworks with different bucket counts are comparable.
STRESS_SCORE = {
    'isc_regime':       {'stable': 0, 'elevated': 1, 'rising': 2, 'distressed': 3},
    'altman_regime':    {'safe': 0, 'grey': 1.5, 'distress': 3},
    'piotroski_regime': {'strong': 0, 'mixed': 1.5, 'weak': 3},
    'beneish_regime':   {'clean': 0, 'manipulator': 3},
    'composite_regime': {'stable': 0, 'elevated': 1, 'rising': 2, 'distressed': 3},
    'outcome_regime':   {'stable': 0, 'elevated': 1, 'rising': 2, 'distressed': 3},
}

FRAMEWORK_LABELS = {
    'isc_regime':       'ISC (Variance EWS)',
    'altman_regime':    'Altman Z',
    'piotroski_regime': 'Piotroski F',
    'beneish_regime':   'Beneish M',
    'composite_regime': 'Composite',
}

# Tiebreak order for "closest-aligned framework" when multiple frameworks tie.
# ISC first because that's the framework being advanced. The point of this is
# transparency about the tiebreak, not gaming results.
TIEBREAK_ORDER = ['isc_regime', 'composite_regime', 'altman_regime',
                  'beneish_regime', 'piotroski_regime']

HORIZONS = [1, 2, 4, 8, 12]
MAX_QUARTERS = 12


# ──────────────────────────────────────────────────────────────────────────────
# Original v1 reports (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def spearman(x, y):
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


def regime_rank(regime, framework):
    order = REGIME_ORDER.get(framework, [])
    if regime in order:
        return order.index(regime)
    return None


def fmt_pct(v, places=1):
    if v is None or pd.isna(v):
        return '   N/A'
    return f"{v * 100:+.{places}f}%"


def make_regime_tables(df, out_path):
    lines = ["=" * 80, "FILTER LAB BACKTEST — REGIME PERFORMANCE TABLES", "=" * 80, ""]
    lines.append("Stocks grouped by predicted regime at Q8; forward stats at each horizon.")
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
                    lines.append(f"  {regime:<14} {0:>5} {'—':>10} {'—':>11} {'—':>10} {'—':>10} {'—':>9}")
                    continue
                mean_ret = bucket['total_return'].mean()
                med_ret = bucket['total_return'].median()
                mean_dd = bucket['max_drawdown'].mean()
                mean_vol = bucket['realized_vol'].mean()
                hit_rate = (bucket['total_return'] > 0).mean()
                lines.append(f"  {regime:<14} {n:>5} {fmt_pct(mean_ret):>10} {fmt_pct(med_ret):>11} "
                             f"{fmt_pct(mean_dd):>10} {fmt_pct(mean_vol):>10} {hit_rate*100:>7.1f}%")
            n_unknown = (sub[framework] == 'unknown').sum()
            if n_unknown > 0:
                lines.append(f"  {'unknown':<14} {n_unknown:>5} (regime not computable)")
        lines.append("")

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def confusion_matrix(df, predicted_col, actual_col='outcome_regime'):
    pred_order = REGIME_ORDER[predicted_col]
    actual_order = REGIME_ORDER[actual_col]
    sub = df[df[predicted_col].isin(pred_order) & df[actual_col].isin(actual_order)]
    if len(sub) == 0:
        return None
    matrix = pd.crosstab(sub[predicted_col], sub[actual_col])
    matrix = matrix.reindex(index=[r for r in pred_order if r in matrix.index],
                            columns=[r for r in actual_order if r in matrix.columns],
                            fill_value=0)
    return matrix


def fmt_confusion(matrix, framework_label):
    if matrix is None or matrix.empty:
        return f"\n  No data for {framework_label}\n"
    lines = []
    n_total = matrix.values.sum()
    lines.append(f"\n  Confusion matrix: {framework_label} predicted (rows) vs realized outcome (cols)")
    lines.append(f"  Total observations: {n_total}\n")
    col_width = 13
    lines.append("  " + " " * 16 + "".join(f"{col:>{col_width}}" for col in matrix.columns))
    lines.append("  " + " " * 16 + "".join("-" * col_width for _ in matrix.columns))
    for pred_regime, row in matrix.iterrows():
        cells = "".join(f"{int(v):>{col_width}}" for v in row.values)
        lines.append(f"  {pred_regime:<16}{cells}")
    return "\n".join(lines)


def compute_lift(df, predicted_col):
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
        return None
    bottom_rate = (bottom['outcome_regime'] == 'distressed').mean()
    lift = bottom_rate / base_rate if base_rate > 0 else None
    return {'worst_pred': worst_pred, 'n_bottom': len(bottom),
            'base_rate': base_rate, 'bottom_rate': bottom_rate, 'lift': lift}


def make_summary(df, out_path):
    lines = ["=" * 80, "FILTER LAB BACKTEST — SUMMARY REPORT", "=" * 80, ""]
    lines.append(f"Total observations (ticker × horizon): {len(df)}")
    lines.append(f"Unique tickers: {df['ticker'].nunique()}")
    lines.append(f"Date range of Q8 observations: {df['obs_date'].min()} to {df['obs_date'].max()}")
    lines.append("")

    # Sample composition
    lines.append("=" * 80)
    lines.append("  SAMPLE COMPOSITION")
    lines.append("=" * 80)
    unique_tickers = df.drop_duplicates('ticker')
    lines.append("  Sector bucket distribution (tickers):")
    for sec, n in unique_tickers['sector_bucket'].value_counts().items():
        lines.append(f"    {sec:<14} {n}")
    for h in HORIZONS:
        sub = df[df['horizon_q'] == h]
        if len(sub) == 0:
            continue
        lines.append(f"\n  {h}Q realized outcome regime distribution (N={len(sub)}):")
        for outcome in REGIME_ORDER['outcome_regime']:
            n = (sub['outcome_regime'] == outcome).sum()
            pct = 100.0 * n / len(sub) if len(sub) > 0 else 0
            lines.append(f"    {outcome:<14} {n:>4} ({pct:.1f}%)")

    # Spearman headline
    lines.append("\n" + "=" * 80)
    lines.append("  HEADLINE: SPEARMAN RANK CORRELATION (predicted regime vs realized)")
    lines.append("=" * 80)
    lines.append("")
    horizons_header = " ".join(f"{h}Q rho".rjust(10) for h in HORIZONS)
    lines.append(f"  {'Framework':<20} {horizons_header}")
    lines.append(f"  {'-'*20} " + " ".join("-" * 10 for _ in HORIZONS))
    headline_results = []
    for framework, label in FRAMEWORK_LABELS.items():
        cells = [f"  {label:<20} "]
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
    lines.append("\n  Best framework per horizon (highest Spearman rho):")
    for h in HORIZONS:
        h_results = [r for r in headline_results if r[2] == h]
        if not h_results:
            continue
        best = max(h_results, key=lambda r: r[3])
        lines.append(f"    {h}Q: {best[1]} (rho = {best[3]:.3f}, N = {best[4]})")

    # Lift
    lines.append("\n" + "=" * 80)
    lines.append("  LIFT IN WORST-PREDICTED-BUCKET")
    lines.append("=" * 80)
    lines.append("\n  Of stocks in the framework's worst-predicted regime, fraction that")
    lines.append("  realized as 'distressed'. Lift = bottom-bucket rate / base rate.\n")
    lines.append(f"  {'Framework':<20} {'Horizon':<8} {'Bottom Pred':<14} {'N':>5} {'Base':>7} {'Bottom':>8} {'Lift':>7}")
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
            lines.append(f"  {label:<20} {h}Q       {r['worst_pred']:<14} {r['n_bottom']:>5} {base:>7} {bot:>8} {lift:>7}")

    # Confusion matrices at 4Q and 8Q
    lines.append("\n" + "=" * 80)
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
# NEW: Per-stock CSV
# ──────────────────────────────────────────────────────────────────────────────

def determine_closest_aligned(row_8q, frameworks):
    """
    For a given ticker (using its 8Q row as the reference), return the framework
    whose stress score is closest to the realized outcome stress score.

    Tiebreak: TIEBREAK_ORDER (ISC first, then Composite, then traditional, then
    Beneish, then Piotroski).

    Returns (framework_key, framework_label, |diff|) or (None, None, None) if
    outcome_regime is unknown.
    """
    outcome = row_8q.get('outcome_regime')
    if not outcome or outcome == 'unknown':
        return None, None, None
    outcome_score = STRESS_SCORE['outcome_regime'].get(outcome)
    if outcome_score is None:
        return None, None, None

    candidates = []
    for fw in frameworks:
        regime = row_8q.get(fw)
        if not regime or regime == 'unknown':
            continue
        score = STRESS_SCORE[fw].get(regime)
        if score is None:
            continue
        diff = abs(score - outcome_score)
        candidates.append((fw, FRAMEWORK_LABELS.get(fw, fw), diff))

    if not candidates:
        return None, None, None

    # Sort by (diff asc, tiebreak index asc)
    def sort_key(c):
        fw = c[0]
        tiebreak_idx = TIEBREAK_ORDER.index(fw) if fw in TIEBREAK_ORDER else len(TIEBREAK_ORDER)
        return (c[2], tiebreak_idx)

    candidates.sort(key=sort_key)
    return candidates[0]


def make_per_stock_csv(df, out_path):
    """
    One row per ticker. Includes all four regimes at Q8, the realized outcomes
    at each horizon, sector-relative returns, max drawdowns, per-quarter returns,
    and which framework was 'closest-aligned' at 8Q.
    """
    frameworks = list(FRAMEWORK_LABELS.keys())
    tickers = df['ticker'].unique()

    rows = []
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker].sort_values('horizon_q')
        if len(ticker_df) == 0:
            continue

        # Use the 8Q row as reference for regime labels (regimes are identical
        # across horizons since they're computed at Q8)
        ref = ticker_df.iloc[0]   # any row works for regime fields
        row_for_alignment = ticker_df[ticker_df['horizon_q'] == 8]
        if len(row_for_alignment) == 0:
            row_for_alignment = ticker_df.iloc[[-1]]
        ref_for_alignment = row_for_alignment.iloc[0].to_dict()

        closest_fw, closest_label, closest_diff = determine_closest_aligned(
            ref_for_alignment, frameworks)

        out_row = {
            'ticker': ticker,
            'sector_bucket': ref['sector_bucket'],
            'sector': ref['sector'],
            'obs_date': ref['obs_date'],
            # Regimes at Q8 (identical across all horizon rows)
            'isc_regime': ref['isc_regime'],
            'isc_score': ref['isc_score'],
            'isc_trend': ref['isc_trend'],
            'isc_ratio': ref['isc_ratio'],
            'altman_regime': ref['altman_regime'],
            'altman_z': ref['altman_z'],
            'piotroski_regime': ref['piotroski_regime'],
            'piotroski_f': ref['piotroski_f'],
            'beneish_regime': ref['beneish_regime'],
            'beneish_m': ref['beneish_m'],
            'composite_regime': ref['composite_regime'],
            # Closest-aligned framework at 8Q
            'closest_aligned_framework_8q': closest_label,
            'closest_aligned_diff_8q': round(closest_diff, 2) if closest_diff is not None else None,
        }
        # Per-horizon outcomes
        for h in HORIZONS:
            h_row = ticker_df[ticker_df['horizon_q'] == h]
            if len(h_row) == 0:
                out_row[f'total_return_{h}q'] = None
                out_row[f'sector_relative_return_{h}q'] = None
                out_row[f'max_drawdown_{h}q'] = None
                out_row[f'outcome_regime_{h}q'] = None
                continue
            hr = h_row.iloc[0]
            out_row[f'total_return_{h}q'] = hr['total_return']
            out_row[f'sector_relative_return_{h}q'] = hr['sector_relative_return']
            out_row[f'max_drawdown_{h}q'] = hr['max_drawdown']
            out_row[f'outcome_regime_{h}q'] = hr['outcome_regime']
        # Per-quarter returns (from any horizon row, they're identical)
        for q in range(1, MAX_QUARTERS + 1):
            col = f'q{q}_return'
            out_row[col] = ref.get(col)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} tickers)")
    return out_df


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Narrative report
# ──────────────────────────────────────────────────────────────────────────────

def fmt_q_returns(per_stock_row):
    """Format the 12 per-quarter returns as a readable trajectory line."""
    parts = []
    for q in range(1, MAX_QUARTERS + 1):
        v = per_stock_row.get(f'q{q}_return')
        if v is None or pd.isna(v):
            parts.append(f"Q+{q}: —")
        else:
            parts.append(f"Q+{q}: {v*100:+.1f}%")
    return "  ".join(parts)


def fmt_sector_rel(v):
    if v is None or pd.isna(v):
        return "N/A"
    sign = "outperformed" if v > 0 else "underperformed"
    return f"{sign} sector peers by {abs(v)*100:.1f}pp"


def narrate_ticker(per_stock_row):
    """Generate a multi-line narrative paragraph for one ticker."""
    t = per_stock_row['ticker']
    sector = per_stock_row.get('sector') or per_stock_row.get('sector_bucket')
    obs_date = per_stock_row['obs_date']

    isc = per_stock_row.get('isc_regime', 'unknown')
    isc_v = per_stock_row.get('isc_score')
    isc_t = per_stock_row.get('isc_trend')
    alt = per_stock_row.get('altman_regime', 'unknown')
    alt_z = per_stock_row.get('altman_z')
    pio = per_stock_row.get('piotroski_regime', 'unknown')
    pio_f = per_stock_row.get('piotroski_f')
    ben = per_stock_row.get('beneish_regime', 'unknown')
    comp = per_stock_row.get('composite_regime', 'unknown')
    closest = per_stock_row.get('closest_aligned_framework_8q', 'N/A')

    ret_4q = per_stock_row.get('total_return_4q')
    ret_8q = per_stock_row.get('total_return_8q')
    ret_12q = per_stock_row.get('total_return_12q')
    dd_8q = per_stock_row.get('max_drawdown_8q')
    dd_12q = per_stock_row.get('max_drawdown_12q')
    secrel_4q = per_stock_row.get('sector_relative_return_4q')
    secrel_8q = per_stock_row.get('sector_relative_return_8q')
    secrel_12q = per_stock_row.get('sector_relative_return_12q')
    outcome_4q = per_stock_row.get('outcome_regime_4q', 'unknown')
    outcome_8q = per_stock_row.get('outcome_regime_8q', 'unknown')
    outcome_12q = per_stock_row.get('outcome_regime_12q', 'unknown')

    lines = []
    lines.append(f"━━━ {t} ({sector}) — Q8 obs date: {obs_date} ━━━")
    lines.append(f"")
    lines.append(f"  At observation (using only data through {obs_date}):")
    lines.append(f"    ISC:        {isc.upper():<14}"
                 + (f"  variance={isc_v:.4f}  trend={isc_t:.2f}" if isc_v is not None and isc_t is not None else ""))
    lines.append(f"    Altman Z:   {alt.upper():<14}"
                 + (f"  Z={alt_z:.2f}" if alt_z is not None else ""))
    lines.append(f"    Piotroski:  {pio.upper():<14}"
                 + (f"  F={pio_f}/9" if pio_f is not None else ""))
    lines.append(f"    Beneish M:  {ben.upper():<14}")
    lines.append(f"    Composite:  {comp.upper():<14}")
    lines.append(f"")
    lines.append(f"  What happened next:")
    if ret_4q is not None:
        lines.append(f"    4Q forward:  {ret_4q*100:+.1f}%  ({fmt_sector_rel(secrel_4q)})  → realized: {outcome_4q.upper()}")
    if ret_8q is not None and dd_8q is not None:
        lines.append(f"    8Q forward:  {ret_8q*100:+.1f}%  max drawdown {dd_8q*100:+.1f}%  ({fmt_sector_rel(secrel_8q)})  → realized: {outcome_8q.upper()}")
    if ret_12q is not None and dd_12q is not None:
        lines.append(f"   12Q forward:  {ret_12q*100:+.1f}%  max drawdown {dd_12q*100:+.1f}%  ({fmt_sector_rel(secrel_12q)})  → realized: {outcome_12q.upper()}")
    lines.append(f"")
    lines.append(f"  Quarter-by-quarter trajectory:")
    # Split into two rows of 6 for readability
    q_line_1 = "    " + "  ".join(
        f"Q+{q}: {(per_stock_row.get(f'q{q}_return') or 0)*100:+5.1f}%"
        if per_stock_row.get(f'q{q}_return') is not None else f"Q+{q}:    —  "
        for q in range(1, 7)
    )
    q_line_2 = "    " + "  ".join(
        f"Q+{q}: {(per_stock_row.get(f'q{q}_return') or 0)*100:+5.1f}%"
        if per_stock_row.get(f'q{q}_return') is not None else f"Q+{q}:    —  "
        for q in range(7, 13)
    )
    lines.append(q_line_1)
    lines.append(q_line_2)
    lines.append(f"")
    lines.append(f"  Closest-aligned framework with realized 8Q outcome: {closest}")
    lines.append(f"  (Note: 'closest-aligned' means the framework whose stress classification")
    lines.append(f"   at Q8 was most consistent with what the stock actually did. This is")
    lines.append(f"   descriptive of past behavior — not a prediction.)")
    lines.append(f"")
    return "\n".join(lines)


def make_narrative_report(per_stock_df, out_path, focus_tickers=None):
    """
    Generate human-readable narratives. By default writes all tickers, grouped
    by ISC regime so the report tells a structured story.

    focus_tickers: optional list — if provided, only narrate these. Otherwise
    a 'highlight' subset is selected for the AISB demo.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("FILTER LAB BACKTEST — PER-STOCK NARRATIVES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Each card shows: regime classifications at Q8, forward outcomes")
    lines.append("at 4Q/8Q/12Q (with sector-relative comparison), quarter-by-quarter")
    lines.append("price trajectory, and which framework was 'closest-aligned' with")
    lines.append("the realized 8Q outcome.")
    lines.append("")
    lines.append("FRAMING: 'Closest-aligned' means the framework whose Q8 classification")
    lines.append("was most consistent with how the stock actually traded. This is")
    lines.append("descriptive of past behavior — not a prediction.")
    lines.append("")

    if focus_tickers:
        sub = per_stock_df[per_stock_df['ticker'].isin(focus_tickers)].copy()
        lines.append(f"Focus tickers: {', '.join(focus_tickers)}")
        lines.append("")
        for _, row in sub.iterrows():
            lines.append(narrate_ticker(row.to_dict()))
        Path(out_path).write_text("\n".join(lines))
        print(f"Wrote {out_path} ({len(sub)} tickers)")
        return

    # Group by ISC regime and write all tickers within each group
    isc_order = REGIME_ORDER['isc_regime']
    for regime in isc_order + ['unknown']:
        sub = per_stock_df[per_stock_df['isc_regime'] == regime].copy()
        if len(sub) == 0:
            continue
        lines.append("=" * 80)
        lines.append(f"  ISC REGIME AT Q8: {regime.upper()}  (N={len(sub)})")
        lines.append("=" * 80)
        lines.append("")
        # Sort by 8Q return descending so best/worst stand out
        sub = sub.sort_values('total_return_8q', ascending=False, na_position='last')
        for _, row in sub.iterrows():
            lines.append(narrate_ticker(row.to_dict()))

    Path(out_path).write_text("\n".join(lines))
    print(f"Wrote {out_path} ({len(per_stock_df)} tickers)")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='backtest_results.csv')
    parser.add_argument('--tables-out', default='backtest_regime_tables.txt')
    parser.add_argument('--summary-out', default='backtest_summary.txt')
    parser.add_argument('--per-stock-out', default='backtest_per_stock.csv')
    parser.add_argument('--narratives-out', default='backtest_narratives.txt')
    parser.add_argument('--focus', default=None,
                        help='Comma-separated tickers for narrative focus (default: all)')
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} rows from {args.results}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    print(f"  Horizons present: {sorted(df['horizon_q'].unique())}")
    print()

    # Original v1 outputs
    make_regime_tables(df, args.tables_out)
    make_summary(df, args.summary_out)

    # New v2 outputs
    per_stock_df = make_per_stock_csv(df, args.per_stock_out)
    focus_list = args.focus.split(',') if args.focus else None
    make_narrative_report(per_stock_df, args.narratives_out, focus_tickers=focus_list)


if __name__ == '__main__':
    main()
