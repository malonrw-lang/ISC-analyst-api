"""
backtest_pipeline.py
====================
Filter Lab walk-forward backtest pipeline.

For each ticker in the snapshot:
  1. Fetch 20Q EDGAR financial history
  2. Score at Q8 using only data through Q8 (ISC, Altman, Piotroski, Beneish, Composite)
  3. Fetch ~5 years of daily prices
  4. Identify observation date (the end-of-quarter date for Q8)
  5. Compute forward stock performance at 2Q, 4Q, 8Q, 12Q horizons
  6. Write one row per (ticker, horizon) to backtest_results.csv

Critical methodology notes:
  - Score is computed using ONLY data up through Q8 (no lookahead).
  - Q8 observation date is the 'end' date of the 8th-most-recent fiscal quarter.
  - Forward return is from obs_date to obs_date + horizon * 91 days.
  - Sector-relative return is stock return minus average return of same
    sector_bucket peers over the same window.

Usage:
  python backtest_pipeline.py --snapshot snapshot.csv [--limit N]

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
HORIZONS_Q = [2, 4, 8, 12]


def observation_date_for_q8(raw):
    """
    Return the calendar date of the end of Q8.

    Q8 = the 8th-oldest quarter in the 20Q window. By convention here, 'first
    8Q' means quarters 1-8 in chronological order. The observation date is the
    'end' index of the 8th quarter — that's the latest point at which the
    scorer is allowed to see data.

    Returns pd.Timestamp or None.
    """
    rev = raw.get('revenue')
    if rev is None or len(rev) < 8:
        return None
    return rev.index[7]   # 0-indexed: position 7 is the 8th observation


def compute_isc_at_q8(prices, obs_date):
    """
    Compute ISC variance regime using only price history up through obs_date.

    Uses the same compute_variance_score() as production, but with prices
    truncated to obs_date.
    """
    if prices is None or obs_date is None:
        return None
    # Truncate price series to obs_date
    truncated = prices[prices.index <= obs_date]
    if len(truncated) < 120:   # need ~6 months minimum
        return None
    return compute_variance_score(truncated, window_days=252, rolling_window=90)


def compute_forward_stats(prices, obs_date, horizon_q):
    """
    Compute forward stock performance from obs_date over horizon_q quarters.

    Returns dict with:
      - total_return: end price / start price - 1
      - ann_return: annualized return
      - max_drawdown: worst peak-to-trough (negative value)
      - realized_vol: annualized stddev of daily log returns
      - n_days: trading days actually covered

    Returns None if insufficient forward data.
    """
    if prices is None or obs_date is None:
        return None

    # Find prices at and after obs_date
    forward = prices[prices.index >= obs_date]
    if len(forward) < 5:
        return None

    horizon_days = horizon_q * TRADING_DAYS_PER_QUARTER
    forward_window = forward.iloc[:horizon_days + 1]   # +1 to include start

    # Need at least 60% of expected window to be meaningful
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

    # Max drawdown
    running_max = forward_window.cummax()
    drawdown_series = (forward_window - running_max) / running_max
    max_dd = float(drawdown_series.min())

    # Realized vol
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
    """
    Bucket the forward outcome into a 4-regime label that mirrors ISC's labels.

    Definitions (thresholds chosen to be roughly symmetric with ISC regime boundaries):
      - stable:     total_return > 0% AND max_drawdown > -15%
      - elevated:   total_return in [-10%, +10%] OR max_drawdown in [-30%, -15%]
      - rising:     total_return in [-25%, -10%] OR max_drawdown in [-50%, -30%]
      - distressed: total_return < -25% OR max_drawdown < -50%

    Worst-bucket wins (if drawdown says distressed but return says elevated, → distressed).
    """
    if forward_stats is None:
        return 'unknown'
    ret = forward_stats.get('total_return')
    dd = forward_stats.get('max_drawdown')
    if ret is None or dd is None:
        return 'unknown'

    # Determine bucket by drawdown
    if dd <= -0.50:
        dd_bucket = 'distressed'
    elif dd <= -0.30:
        dd_bucket = 'rising'
    elif dd <= -0.15:
        dd_bucket = 'elevated'
    else:
        dd_bucket = 'stable'

    # Determine bucket by return
    if ret <= -0.25:
        ret_bucket = 'distressed'
    elif ret <= -0.10:
        ret_bucket = 'rising'
    elif ret <= 0.10:
        ret_bucket = 'elevated'
    else:
        ret_bucket = 'stable'

    # Worst wins
    order = {'stable': 0, 'elevated': 1, 'rising': 2, 'distressed': 3}
    if order[dd_bucket] >= order[ret_bucket]:
        return dd_bucket
    return ret_bucket


# ──────────────────────────────────────────────────────────────────────────────
# Per-ticker pipeline
# ──────────────────────────────────────────────────────────────────────────────

def process_ticker(ticker, snapshot_row, verbose=False):
    """
    Run the full backtest for one ticker.

    Returns list of dicts (one per horizon) ready for CSV writing, or empty list
    on failure.
    """
    rows = []

    # 1. CIK lookup
    cik = snapshot_row.get('cik')
    if not cik or pd.isna(cik):
        cik = get_cik(ticker)
    if not cik:
        if verbose:
            print(f"  {ticker}: no CIK")
        return rows
    # Ensure 10-digit zero-padded
    try:
        cik = str(int(cik)).zfill(10)
    except (TypeError, ValueError):
        cik = str(cik).zfill(10)

    # 2. EDGAR facts
    facts = get_facts(cik)
    if not facts:
        if verbose:
            print(f"  {ticker}: no EDGAR facts")
        return rows

    # 3. Extract 20Q raw series
    raw = {}
    for key in TAG_MAP:
        raw[key] = extract_series(facts, key, n=20)

    rev = raw.get('revenue')
    if rev is None or len(rev) < 12:
        if verbose:
            print(f"  {ticker}: insufficient revenue history ({len(rev) if rev is not None else 0}Q)")
        return rows

    # 4. Determine obs_q. We want at least 8Q lookback. Use Q8 in the
    # chronological series (position 7, 0-indexed).
    obs_q = 8
    obs_date = observation_date_for_q8(raw)
    if obs_date is None:
        if verbose:
            print(f"  {ticker}: no Q8 observation date")
        return rows

    # 5. Score at obs_q using only data through Q8
    altman = altman_z_at(raw, obs_q)
    alt_reg = altman_regime(altman)

    f_score, f_signals = piotroski_at(raw, obs_q)
    pio_reg = piotroski_regime(f_score)

    m_score = beneish_at(raw, obs_q)
    ben_reg = beneish_regime(m_score)

    # 6. Fetch prices and compute ISC at Q8
    prices, price_source = fetch_daily_prices(ticker, days=1825)   # ~5y
    if prices is None or len(prices) < 200:
        if verbose:
            print(f"  {ticker}: no price data ({price_source})")
        return rows

    isc = compute_isc_at_q8(prices, obs_date)
    if isc and 'error' not in isc:
        isc_score = isc.get('mean_variance')
        isc_trend = isc.get('variance_trend')
        isc_ratio = isc.get('variance_ratio')
        isc_reg = isc.get('regime')
    else:
        isc_score = isc_trend = isc_ratio = None
        isc_reg = 'unknown'

    composite_reg = composite_regime(isc_reg, alt_reg, pio_reg, ben_reg)

    # 7. Sector context (from snapshot, falls back to EDGAR submissions)
    sector_bucket = snapshot_row.get('sector_bucket')
    if not sector_bucket or pd.isna(sector_bucket):
        submissions = get_submissions(cik)
        sic = get_company_sic(submissions) if submissions else None
        sector_bucket = detect_sector(sic)
    sector_text = snapshot_row.get('sector', '')

    # 8. Forward stats at each horizon
    for h in HORIZONS_Q:
        fwd = compute_forward_stats(prices, obs_date, h)
        if fwd is None:
            continue
        outcome_reg = realized_outcome_regime(fwd)

        rows.append({
            'ticker': ticker,
            'sector_bucket': sector_bucket,
            'sector': sector_text,
            'obs_date': str(obs_date.date()),
            'horizon_q': h,
            # Scores
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
            # Forward
            'total_return': fwd['total_return'],
            'ann_return': fwd['ann_return'],
            'max_drawdown': fwd['max_drawdown'],
            'realized_vol': fwd['realized_vol'],
            'n_forward_days': fwd['n_days'],
            'outcome_regime': outcome_reg,
            # Meta
            'price_source': price_source,
        })

    if verbose and rows:
        # Print the 8Q row for visual sanity check
        row_8q = next((r for r in rows if r['horizon_q'] == 8), rows[0])
        print(f"  {ticker}: obs={row_8q['obs_date']} ISC={row_8q['isc_regime']:>10} "
              f"Alt={row_8q['altman_regime']:>8} Pio={row_8q['piotroski_regime']:>6} "
              f"→ ret_8q={row_8q['total_return']:+.2%} dd={row_8q['max_drawdown']:+.2%} "
              f"outcome={row_8q['outcome_regime']}")
    elif verbose:
        print(f"  {ticker}: no forward stats")

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='snapshot.csv',
                        help='Path to snapshot CSV')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit to first N tickers (for smoke testing)')
    parser.add_argument('--out', default='backtest_results.csv',
                        help='Output CSV path')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print per-ticker progress')
    args = parser.parse_args()

    # Load snapshot
    snap = pd.read_csv(args.snapshot)
    print(f"Loaded {len(snap)} tickers from {args.snapshot}")
    if args.limit:
        snap = snap.iloc[:args.limit]
        print(f"Limited to first {args.limit} tickers")

    # Prepare output
    out_path = Path(args.out)
    fieldnames = [
        'ticker', 'sector_bucket', 'sector', 'obs_date', 'horizon_q',
        'isc_score', 'isc_trend', 'isc_ratio', 'isc_regime',
        'altman_z', 'altman_regime',
        'piotroski_f', 'piotroski_regime',
        'beneish_m', 'beneish_regime',
        'composite_regime',
        'total_return', 'ann_return', 'max_drawdown', 'realized_vol',
        'n_forward_days', 'outcome_regime', 'price_source',
    ]

    n_success = 0
    n_failed = 0
    n_rows = 0
    start_time = time.time()

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, snapshot_row in snap.iterrows():
            ticker = snapshot_row['ticker']
            if pd.isna(ticker):
                continue
            ticker = str(ticker).strip().upper()

            elapsed = time.time() - start_time
            print(f"[{idx+1}/{len(snap)}] {ticker} (elapsed {elapsed:.0f}s, {n_success} OK, {n_failed} fail, {n_rows} rows)")

            try:
                rows = process_ticker(ticker, snapshot_row.to_dict(), verbose=args.verbose)
                if rows:
                    for r in rows:
                        writer.writerow(r)
                    n_rows += len(rows)
                    n_success += 1
                else:
                    n_failed += 1
            except Exception as e:
                n_failed += 1
                print(f"  {ticker}: EXCEPTION {type(e).__name__}: {e}")

            # Tiingo rate limit: pause every 50 calls
            if (idx + 1) % 50 == 0:
                time.sleep(2)

    elapsed = time.time() - start_time
    print(f"\nDone. {n_success} tickers succeeded, {n_failed} failed, {n_rows} total rows.")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Output: {out_path}")


if __name__ == '__main__':
    main()
