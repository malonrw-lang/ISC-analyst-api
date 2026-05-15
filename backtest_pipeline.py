"""
backtest_pipeline.py — v3 (rolling obs_q)
==========================================
Filter Lab walk-forward backtest pipeline.

CHANGES from v2:
  - ROLLING OBS_Q: each ticker now generates rows at multiple observation
    quarters: obs_q ∈ [8, 10, 12, 14, 16]. Previously every ticker was
    pinned to obs_q=8 (earliest valid observation). Now we get up to 5
    observations per ticker — roughly 5× the sample size.
  - Generalized observation_date_for_q8(raw) → observation_date_at(raw, obs_q)
  - Added obs_q column to CSV output. Each (ticker, obs_q, horizon) is now a
    unique row. ~2,400 total rows expected for 478-ticker universe (vs ~1,400 before).
  - Sector-relative returns now compute peer means within (sector, obs_q, horizon)
    instead of (sector, horizon), so peers are time-matched and the relative-
    return signal isn't contaminated by obs_q clustering.
  - Skips obs_q values that don't have enough forward price data (e.g. a 20Q
    history ticker with obs_q=16 has only 4Q forward window, not enough for
    the 12Q horizon).

For each ticker in the snapshot, for each valid obs_q in OBS_Q_VALUES:
  1. Fetch 20Q EDGAR financial history
  2. Score at obs_q using only data through obs_q (ISC, Altman, Piotroski, Beneish, Composite)
  3. Fetch ~5 years of daily prices
  4. Identify observation date (the end-of-quarter date for that obs_q)
  5. Compute forward stock performance at 1Q, 2Q, 4Q, 8Q, 12Q horizons
  6. Decompose return into per-quarter chunks
  7. Skip obs_q if forward price data is insufficient
  After loop: compute sector means per (obs_q, horizon), derive sector_relative_return

Usage:
  python backtest_pipeline.py --snapshot snapshot.csv [--limit N]
  python backtest_pipeline.py --snapshot snapshot.csv --obs-q 8,12,16

Author: Ryan W. Malone
"""
import sys
import os
import argparse
import time
import csv
from pathlib import Path

import numpy as np
import pandas as pd

# Imports from the main ISC-analyst-api codebase
from main import (
    get_cik, get_facts, get_submissions, detect_sector, get_company_sic,
    extract_series, TAG_MAP,
)
from price_data import fetch_daily_prices
from variance_score import compute_variance_score

from backtest_scorer import (
    altman_z_at, altman_regime,
    piotroski_at, piotroski_regime,
    beneish_at, beneish_regime,
    composite_regime,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_QUARTER = 63   # ≈ 252/4
HORIZONS_Q = [1, 2, 4, 8, 12]
MAX_QUARTERS = max(HORIZONS_Q)   # 12 — number of per-quarter columns

# Rolling observation quarters. Each ticker is scored at every obs_q in this
# list where its EDGAR history is long enough AND its forward price window
# is long enough to compute the 12Q horizon.
#   - obs_q must be ≥ 8 (scorer requires 8 quarters of look-back for Beneish)
#   - obs_q + 12 ≤ EDGAR history length (need 12Q forward fundamental context)
#   - obs_date + 12 quarters ≤ price data end (need forward price window)
DEFAULT_OBS_Q_VALUES = [8, 10, 12, 14, 16]


def observation_date_at(raw, obs_q):
    """Return the calendar date of the end of obs_q (1-indexed quarter)."""
    rev = raw.get('revenue')
    if rev is None or len(rev) < obs_q:
        return None
    return rev.index[obs_q - 1]


def compute_isc_at(prices, obs_date):
    """ISC variance regime using only price history through obs_date.
    Renamed from compute_isc_at_q8 — now works at any obs_date."""
    if prices is None or obs_date is None:
        return None
    truncated = prices[prices.index <= obs_date]
    if len(truncated) < 120:
        return None
    return compute_variance_score(truncated, window_days=252, rolling_window=90)


def compute_per_quarter_returns(prices, obs_date, n_quarters=MAX_QUARTERS):
    """
    Decompose forward price action into per-quarter return chunks.

    Returns a list of n_quarters floats. Each entry is the return from the
    start of that quarter to its end. Missing quarters are None.

    Q+1 = obs_date to obs_date + 63 trading days
    Q+2 = +63 to +126 trading days
    ...
    """
    out = [None] * n_quarters
    if prices is None or obs_date is None:
        return out

    forward = prices[prices.index >= obs_date]
    if len(forward) < 5:
        return out

    for q in range(n_quarters):
        start_idx = q * TRADING_DAYS_PER_QUARTER
        end_idx = (q + 1) * TRADING_DAYS_PER_QUARTER
        if end_idx >= len(forward):
            # Partial quarter: only count if >60% of quarter available
            if start_idx >= len(forward) - 5:
                continue
            available = len(forward) - 1
            if available - start_idx < TRADING_DAYS_PER_QUARTER * 0.6:
                continue
            end_idx = available
        if start_idx >= len(forward):
            continue
        start_price = float(forward.iloc[start_idx])
        end_price = float(forward.iloc[end_idx])
        if start_price <= 0 or np.isnan(start_price) or np.isnan(end_price):
            continue
        out[q] = round((end_price / start_price) - 1.0, 4)
    return out


def compute_forward_stats(prices, obs_date, horizon_q):
    """Forward stock performance over horizon_q quarters from obs_date."""
    if prices is None or obs_date is None:
        return None

    forward = prices[prices.index >= obs_date]
    if len(forward) < 5:
        return None

    horizon_days = horizon_q * TRADING_DAYS_PER_QUARTER
    forward_window = forward.iloc[:horizon_days + 1]

    if len(forward_window) < horizon_days * 0.6:
        return None

    start_price = float(forward_window.iloc[0])
    end_price = float(forward_window.iloc[-1])
    if start_price <= 0 or np.isnan(start_price):
        return None

    total_return = (end_price / start_price) - 1.0
    n_days = len(forward_window) - 1
    years = n_days / 252.0
    ann_return = (1 + total_return) ** (1.0 / years) - 1.0 if years > 0 else None

    running_max = forward_window.cummax()
    drawdown_series = (forward_window - running_max) / running_max
    max_dd = float(drawdown_series.min())

    log_returns = np.log(forward_window / forward_window.shift(1)).dropna()
    realized_vol = float(log_returns.std() * np.sqrt(252)) if len(log_returns) >= 10 else None

    return {
        'total_return': round(total_return, 4),
        'ann_return': round(ann_return, 4) if ann_return is not None else None,
        'max_drawdown': round(max_dd, 4),
        'realized_vol': round(realized_vol, 4) if realized_vol is not None else None,
        'n_days': n_days,
    }


def realized_outcome_regime(forward_stats):
    """4-bucket realized outcome label."""
    if forward_stats is None:
        return 'unknown'
    ret = forward_stats.get('total_return')
    dd = forward_stats.get('max_drawdown')
    if ret is None or dd is None:
        return 'unknown'

    if dd <= -0.50:
        dd_bucket = 'distressed'
    elif dd <= -0.30:
        dd_bucket = 'rising'
    elif dd <= -0.15:
        dd_bucket = 'elevated'
    else:
        dd_bucket = 'stable'

    if ret <= -0.25:
        ret_bucket = 'distressed'
    elif ret <= -0.10:
        ret_bucket = 'rising'
    elif ret <= 0.10:
        ret_bucket = 'elevated'
    else:
        ret_bucket = 'stable'

    order = {'stable': 0, 'elevated': 1, 'rising': 2, 'distressed': 3}
    if order[dd_bucket] >= order[ret_bucket]:
        return dd_bucket
    return ret_bucket


# ──────────────────────────────────────────────────────────────────────────────
# Per-ticker, per-obs_q pipeline
# ──────────────────────────────────────────────────────────────────────────────

def score_at_obs_q(ticker, raw, prices, snapshot_row, obs_q,
                   sector_bucket, sector_text, price_source, verbose=False):
    """
    Run framework scoring + forward-stats at one obs_q.
    Returns a list of rows (one per horizon) or [] if obs_q is invalid for
    this ticker (insufficient EDGAR or insufficient forward price data).
    """
    rows = []

    # Check we have enough EDGAR history for this obs_q
    rev = raw.get('revenue')
    if rev is None or len(rev) < obs_q:
        if verbose:
            print(f"  {ticker} @ obs_q={obs_q}: insufficient EDGAR history "
                  f"({len(rev) if rev is not None else 0}Q)")
        return rows

    obs_date = observation_date_at(raw, obs_q)
    if obs_date is None:
        return rows

    # Check we have enough forward price data for the 12Q horizon
    # (Otherwise this obs_q is wasted — most horizons won't compute)
    if prices is not None:
        forward = prices[prices.index >= obs_date]
        min_forward_days = MAX_QUARTERS * TRADING_DAYS_PER_QUARTER * 0.6
        if len(forward) < min_forward_days:
            if verbose:
                print(f"  {ticker} @ obs_q={obs_q}: only {len(forward)}d forward "
                      f"price data (need {min_forward_days:.0f}d for 12Q horizon)")
            return rows

    # Score frameworks at this obs_q
    altman = altman_z_at(raw, obs_q)
    alt_reg = altman_regime(altman)

    f_score, f_signals = piotroski_at(raw, obs_q)
    pio_reg = piotroski_regime(f_score)

    m_score = beneish_at(raw, obs_q)
    ben_reg = beneish_regime(m_score)

    isc = compute_isc_at(prices, obs_date)
    if isc and 'error' not in isc:
        isc_score = isc.get('mean_variance')
        isc_trend = isc.get('variance_trend')
        isc_ratio = isc.get('variance_ratio')
        isc_reg = isc.get('regime')
    else:
        isc_score = isc_trend = isc_ratio = None
        isc_reg = 'unknown'

    composite_reg = composite_regime(isc_reg, alt_reg, pio_reg, ben_reg)

    # Per-quarter returns — once per (ticker, obs_q), broadcast across horizons
    quarter_returns = compute_per_quarter_returns(prices, obs_date, MAX_QUARTERS)

    for h in HORIZONS_Q:
        fwd = compute_forward_stats(prices, obs_date, h)
        if fwd is None:
            continue
        outcome_reg = realized_outcome_regime(fwd)

        row = {
            'ticker': ticker,
            'sector_bucket': sector_bucket,
            'sector': sector_text,
            'obs_date': str(obs_date.date()),
            'obs_q': obs_q,
            'horizon_q': h,
            'isc_score': isc_score,
            'isc_trend': isc_trend,
            'isc_ratio': isc_ratio,
            'isc_regime': isc_reg,
            'altman_z': altman,
            'altman_regime': alt_reg,
            'piotroski_f': f_score,
            'piotroski_regime': pio_reg,
            'beneish_m': m_score,
            'beneish_regime': ben_reg,
            'composite_regime': composite_reg,
            'total_return': fwd['total_return'],
            'ann_return': fwd['ann_return'],
            'max_drawdown': fwd['max_drawdown'],
            'realized_vol': fwd['realized_vol'],
            'n_forward_days': fwd['n_days'],
            'outcome_regime': outcome_reg,
            'price_source': price_source,
            'sector_relative_return': None,
        }
        for q in range(MAX_QUARTERS):
            row[f'q{q+1}_return'] = quarter_returns[q]
        rows.append(row)

    if verbose and rows:
        row_8q = next((r for r in rows if r['horizon_q'] == 8), rows[0])
        print(f"  {ticker} @ obs_q={obs_q} ({row_8q['obs_date']}): "
              f"ISC={row_8q['isc_regime']:>10} Alt={row_8q['altman_regime']:>8} "
              f"→ 8Q ret={row_8q['total_return']:+.2%} "
              f"outcome={row_8q['outcome_regime']}")

    return rows


def process_ticker(ticker, snapshot_row, obs_q_values, verbose=False):
    """Run the full backtest for one ticker across all valid obs_q values."""
    cik = snapshot_row.get('cik')
    if not cik or pd.isna(cik):
        cik = get_cik(ticker)
    if not cik:
        if verbose:
            print(f"  {ticker}: no CIK")
        return []
    try:
        cik = str(int(cik)).zfill(10)
    except (TypeError, ValueError):
        cik = str(cik).zfill(10)

    facts = get_facts(cik)
    if not facts:
        if verbose:
            print(f"  {ticker}: no EDGAR facts")
        return []

    raw = {}
    for key in TAG_MAP:
        raw[key] = extract_series(facts, key, n=20)

    rev = raw.get('revenue')
    if rev is None or len(rev) < 12:
        if verbose:
            print(f"  {ticker}: insufficient revenue history ({len(rev) if rev is not None else 0}Q)")
        return []

    # Fetch sector once per ticker
    sector_bucket = snapshot_row.get('sector_bucket')
    if not sector_bucket or pd.isna(sector_bucket):
        submissions = get_submissions(cik)
        sic = get_company_sic(submissions) if submissions else None
        sector_bucket = detect_sector(sic)
    sector_text = snapshot_row.get('sector', '')

    # Fetch prices once per ticker — same 5-year window covers all obs_q values
    prices, price_source = fetch_daily_prices(ticker, days=1825)
    if prices is None or len(prices) < 200:
        if verbose:
            print(f"  {ticker}: no price data ({price_source})")
        return []

    # Loop over obs_q values — score at each valid one
    all_rows = []
    for obs_q in obs_q_values:
        obs_rows = score_at_obs_q(
            ticker, raw, prices, snapshot_row, obs_q,
            sector_bucket, sector_text, price_source, verbose=verbose
        )
        all_rows.extend(obs_rows)

    if verbose and not all_rows:
        print(f"  {ticker}: no valid obs_q produced rows")

    return all_rows


# ──────────────────────────────────────────────────────────────────────────────
# Sector-relative computation (second pass) — now grouped by obs_q too
# ──────────────────────────────────────────────────────────────────────────────

def add_sector_relative_returns(all_rows):
    """
    Compute sector_relative_return = total_return - mean(sector peers @ same obs_q, horizon).
    Modifies rows in place.

    GROUPING CHANGE from v2: now groups by (sector, obs_q, horizon) so peers
    are time-matched. Previously grouped by (sector, horizon) which silently
    mixed obs_q cohorts.
    """
    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return
    sector_means = df.groupby(['sector_bucket', 'obs_q', 'horizon_q'])['total_return'].mean().to_dict()

    print("\nSector mean returns by (sector, obs_q, horizon):")
    print("(Showing 8Q horizon only for brevity)")
    for (sec, oq, h), m in sorted(sector_means.items()):
        if h != 8:
            continue
        n = ((df['sector_bucket'] == sec) &
             (df['obs_q'] == oq) &
             (df['horizon_q'] == h)).sum()
        print(f"  {sec:<14} obs_q={oq:>2} {h:>2}Q  N={n:>4}  mean={m*100:+7.2f}%")

    for row in all_rows:
        key = (row['sector_bucket'], row['obs_q'], row['horizon_q'])
        sector_mean = sector_means.get(key)
        if sector_mean is not None and row['total_return'] is not None:
            row['sector_relative_return'] = round(row['total_return'] - sector_mean, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='snapshot.csv')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', default='backtest_results.csv')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--obs-q', default=None,
                        help='Comma-separated obs_q values to score at (default: '
                             f'{",".join(map(str, DEFAULT_OBS_Q_VALUES))})')
    args = parser.parse_args()

    # Parse obs_q values
    if args.obs_q:
        obs_q_values = [int(x.strip()) for x in args.obs_q.split(',') if x.strip()]
    else:
        obs_q_values = DEFAULT_OBS_Q_VALUES

    print(f"Scoring at obs_q values: {obs_q_values}")
    print(f"(Each ticker can produce up to {len(obs_q_values)} × {len(HORIZONS_Q)} "
          f"= {len(obs_q_values) * len(HORIZONS_Q)} rows if all obs_q valid)")

    snap = pd.read_csv(args.snapshot)
    print(f"Loaded {len(snap)} tickers from {args.snapshot}")
    if args.limit:
        snap = snap.iloc[:args.limit]
        print(f"Limited to first {args.limit} tickers")

    fieldnames = [
        'ticker', 'sector_bucket', 'sector', 'obs_date', 'obs_q', 'horizon_q',
        'isc_score', 'isc_trend', 'isc_ratio', 'isc_regime',
        'altman_z', 'altman_regime',
        'piotroski_f', 'piotroski_regime',
        'beneish_m', 'beneish_regime',
        'composite_regime',
        'total_return', 'ann_return', 'max_drawdown', 'realized_vol',
        'sector_relative_return',
        'n_forward_days', 'outcome_regime', 'price_source',
    ] + [f'q{q+1}_return' for q in range(MAX_QUARTERS)]

    all_rows = []
    n_success = 0
    n_failed = 0
    start_time = time.time()

    for idx, snapshot_row in snap.iterrows():
        ticker = snapshot_row['ticker']
        if pd.isna(ticker):
            continue
        ticker = str(ticker).strip().upper()

        elapsed = time.time() - start_time
        print(f"[{idx+1}/{len(snap)}] {ticker} (elapsed {elapsed:.0f}s, "
              f"{n_success} OK, {n_failed} fail, {len(all_rows)} rows)")

        try:
            rows = process_ticker(ticker, snapshot_row.to_dict(),
                                  obs_q_values, verbose=args.verbose)
            if rows:
                all_rows.extend(rows)
                n_success += 1
            else:
                n_failed += 1
        except Exception as e:
            n_failed += 1
            print(f"  {ticker}: EXCEPTION {type(e).__name__}: {e}")

        if (idx + 1) % 50 == 0:
            time.sleep(2)

    print(f"\nComputing sector-relative returns over {len(all_rows)} rows...")
    add_sector_relative_returns(all_rows)

    out_path = Path(args.out)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            for q in range(MAX_QUARTERS):
                row.setdefault(f'q{q+1}_return', None)
            writer.writerow(row)

    elapsed = time.time() - start_time
    print(f"\nDone. {n_success} tickers succeeded, {n_failed} failed, "
          f"{len(all_rows)} total rows.")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {out_path}")


if __name__ == '__main__':
    main()
