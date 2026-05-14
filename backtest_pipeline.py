#!/usr/bin/env python3
"""
THE FILTER LAB — Walk-Forward Backtest Pipeline (Phase 2 prototype)
====================================================================

What this does
--------------
Tests the central claim of the IRFA paper:
    Structural early-warning signals computed from financial statements
    discriminate future price behavior with greater power than traditional
    fundamental metrics.

Methodology
-----------
For each ticker in the S&P 500 snapshot:
  1. Pull 20Q of EDGAR data + Tiingo daily prices (uses existing pipeline)
  2. Identify the OBSERVATION POINT — the date that ends quarter 8 of the
     20Q EDGAR window (so observation has 8Q of history available)
  3. Compute scores AS OF observation date using ONLY data through that date:
       - ISC variance EWS regime
       - Altman Z
       - Beneish M-Score (where sector-eligible)
       - Piotroski F
       - Composite (≥2 of 3 lenses agree)
  4. Look forward 2Q / 4Q / 8Q / 12Q from observation date
  5. Record realized log return and drawdown over each lookforward horizon
  6. Aggregate: regime → outcome distribution

Output
------
  results/backtest_observations.csv      — one row per ticker
  results/backtest_summary.txt           — horse race AUC table
  results/backtest_distressed_cohort.csv — ISC distressed list at observation

How to run
----------
  # smoke test with 10 tickers (takes 2-3 minutes)
  python3 backtest_pipeline.py --limit 10

  # full S&P 500 run (takes 30-60 min depending on Tiingo throttling)
  python3 backtest_pipeline.py

  # custom ticker list
  python3 backtest_pipeline.py --tickers AAPL MSFT GOOGL NVDA

Honest caveats
--------------
1. SURVIVORSHIP BIAS — Snapshot only contains tickers that exist today.
   Companies that went bankrupt 2022-2025 are missing. This BIASES THE
   TEST AGAINST the framework's hypothesis (worst outcomes excluded).
   If signal persists despite this, real effect is stronger than measured.

2. OBSERVATION DATE VARIES BY TICKER — Each ticker's observation date is
   the end of their 8th-earliest quarter. For most S&P 500 names with full
   20Q history, this lands roughly 2021-2022. Lookforward of 12Q lands
   2024-2025. Final results aggregate across this calendar dispersion.

3. NO REGIME RETRAINING — Threshold rules (Z<1.81, M>-1.78, etc.) are
   canonical published values. ISC thresholds (variance EWS regime
   classifier) come from the live pipeline as-of-the-observation logic.

4. PROFITABILITY ATTRIBUTION — Outcome metrics are price-based only.
   Dividends not reinvested. Total return ≠ price return for dividend
   stocks; treat results as price-discrimination not total-return-prediction.

Dependencies
------------
  - main.py (existing pipeline)
  - variance_score.py
  - price_data.py
  - pandas, numpy
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Import existing pipeline modules. main.py exposes the scoring functions and
# data extraction helpers we need.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Pure scoring functions — date-agnostic, take raw values
    from main import (
        ttm,
        aligned_ttm_ratio,
        is_trend_up,
        safe_div,
        compute_altman_z,
        compute_piotroski,
        compute_beneish_m_score,
        extract_series,
        extract_series_annual,
        TAG_MAP,
        FLOW_KEYS,
    )
    # Data fetchers
    from price_data import fetch_daily_prices, fetch_basic_market_metadata
    # Variance EWS — takes a price series, returns regime classification
    from variance_score import compute_variance_score
except ImportError as e:
    sys.stderr.write(f"Could not import pipeline modules: {e}\n")
    sys.stderr.write("This script must be run from the same directory as main.py.\n")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
SNAPSHOT_CSV = Path('2026-05-14.csv')  # source of tickers + CIKs
RESULTS_DIR = Path('backtest_results')
RESULTS_DIR.mkdir(exist_ok=True)

# Quarters into the 20Q window where observation point sits.
# OBS_QUARTER=8 means "after 8 quarters of EDGAR data is available".
OBS_QUARTER = 8

# Forward horizons in quarters (1Q ≈ 63 trading days)
LOOKFORWARD_HORIZONS_Q = [2, 4, 8, 12]

# Minimum number of EDGAR quarters required (need OBS_QUARTER for observation,
# plus enough lookforward in price data — the latter is checked at price layer).
MIN_QUARTERS_REQUIRED = OBS_QUARTER + 4  # need at least 4 quarters past obs

# Variance EWS rolling window matches production
VARIANCE_WINDOW_DAYS = 252      # 1 trading year
VARIANCE_ROLLING_WINDOW = 90    # 90-day window for regime classification

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('backtest')


# ─────────────────────────────────────────────────────────────────────
# EDGAR + scoring at a historical observation point
# ─────────────────────────────────────────────────────────────────────
def fetch_company_facts(cik):
    """Fetch full company facts JSON from SEC EDGAR. Returns dict or None."""
    import urllib.request
    cik_padded = str(cik).zfill(10)
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json'
    headers = {
        'User-Agent': 'TheFilterLab Research backtest malonrw@gmail.com',
        'Accept-Encoding': 'gzip, deflate',
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            if resp.headers.get('Content-Encoding') == 'gzip':
                import gzip
                data = gzip.decompress(data)
            return json.loads(data)
    except Exception as e:
        log.warning(f'EDGAR fetch failed for CIK {cik}: {e}')
        return None


def slice_to_observation_point(series, obs_q=OBS_QUARTER):
    """Take a pandas Series of quarterly observations and return only the
    first `obs_q` quarters. Used to simulate "as-of" the observation date."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    # Series is sorted oldest → newest. First obs_q entries = pre-observation.
    return series.iloc[:obs_q]


def compute_observation_date(raw_series_dict, obs_q=OBS_QUARTER):
    """Find the date corresponding to the end of quarter OBS_Q. We use the
    revenue series as the anchor since it's the most reliable timestamp."""
    rev = raw_series_dict.get('revenue')
    if rev is None or len(rev) < obs_q:
        return None
    # Series index should be dates. Take the index of the obs_q-th entry.
    try:
        obs_date = rev.index[obs_q - 1]
        # Convert pandas Timestamp to date
        if isinstance(obs_date, pd.Timestamp):
            return obs_date.date()
        return obs_date
    except Exception:
        return None


def build_historical_raw(facts, obs_q=OBS_QUARTER):
    """Extract EDGAR series sliced to the first OBS_Q quarters.
    Returns dict of pandas Series keyed by metric name."""
    if not facts:
        return None

    # The same keys main.py extracts
    metric_keys = [
        'revenue', 'cost_of_revenue', 'gross_profit',
        'operating_income', 'net_income', 'ebitda', 'da',
        'interest_expense', 'cfo', 'cfi', 'cff', 'capex',
        'total_assets', 'total_liabilities', 'total_equity',
        'cash', 'total_debt', 'long_term_debt',
        'current_assets', 'current_liabilities',
        'retained_earnings', 'shares_outstanding',
    ]

    raw = {}
    for k in metric_keys:
        full = extract_series(facts, k, n=20)
        raw[k] = slice_to_observation_point(full, obs_q)

    return raw


def compute_scores_at_observation(raw, obs_q=OBS_QUARTER):
    """Run all four scoring systems on the sliced raw data.
    Returns dict with regime classifications + raw scores."""
    scores = {
        'isc_regime': None,
        'altman_z': None,
        'altman_class': None,
        'piotroski_f': None,
        'piotroski_class': None,
        'beneish_m': None,
        'beneish_class': None,
        'composite_class': None,
    }

    # --- TTM values for Altman Z inputs ---
    rev_ttm = ttm(raw.get('revenue'))
    oi_ttm = ttm(raw.get('operating_income'))
    ta = raw.get('total_assets').iloc[-1] if len(raw.get('total_assets', [])) else None
    re = raw.get('retained_earnings').iloc[-1] if len(raw.get('retained_earnings', [])) else None
    tl = raw.get('total_liabilities').iloc[-1] if len(raw.get('total_liabilities', [])) else None
    ca = raw.get('current_assets').iloc[-1] if len(raw.get('current_assets', [])) else None
    cl = raw.get('current_liabilities').iloc[-1] if len(raw.get('current_liabilities', [])) else None

    # market cap at observation: approximate using shares × close price at obs date.
    # we don't have it precisely here; will be filled in from the price data layer.
    # For now skip the mktcap-sensitive Z component by passing None and letting compute_altman_z handle it.

    if all(v is not None for v in [ta, re, oi_ttm, rev_ttm, tl, ca, cl]):
        try:
            z = compute_altman_z(ta, re, oi_ttm, rev_ttm, tl, ca, cl, None)
            scores['altman_z'] = z
            if z is not None:
                if z < 1.81: scores['altman_class'] = 'distressed'
                elif z < 3.0: scores['altman_class'] = 'grey'
                else: scores['altman_class'] = 'safe'
        except Exception as e:
            log.debug(f'altman_z compute failed: {e}')

    # --- Piotroski F ---
    cfo_ttm = ttm(raw.get('cfo'))
    ni_ttm = ttm(raw.get('net_income'))
    roa = safe_div(ni_ttm, ta) if (ni_ttm is not None and ta) else None

    if all(v is not None for v in [roa, cfo_ttm, ni_ttm]):
        try:
            f, _ = compute_piotroski(
                roa, cfo_ttm, ni_ttm,
                is_trend_up(raw.get('net_income')),
                is_trend_up(raw.get('long_term_debt')) == False,
                is_trend_up(raw.get('current_assets')),
                is_trend_up(raw.get('shares_outstanding')),
                is_trend_up(raw.get('gross_profit')),
                is_trend_up(raw.get('revenue')),
            )
            scores['piotroski_f'] = f
            if f is not None:
                if f <= 3: scores['piotroski_class'] = 'weak'
                elif f <= 6: scores['piotroski_class'] = 'mixed'
                else: scores['piotroski_class'] = 'strong'
        except Exception as e:
            log.debug(f'piotroski compute failed: {e}')

    # --- Beneish M-Score (annual) ---
    # Beneish uses annual data; the 8Q observation point doesn't map perfectly.
    # We compute it from the latest 2 annual data points available at obs.
    # For simplicity, deferring Beneish to a follow-up. Set null for now.
    # TODO: implement Beneish from extract_series_annual sliced to obs period.

    # --- Composite vote ---
    stress_votes = 0
    clean_votes = 0
    n_voters = 0
    if scores['altman_class']:
        n_voters += 1
        if scores['altman_class'] == 'distressed': stress_votes += 1
        elif scores['altman_class'] == 'safe': clean_votes += 1
    if scores['piotroski_class']:
        n_voters += 1
        if scores['piotroski_class'] == 'weak': stress_votes += 1
        elif scores['piotroski_class'] == 'strong': clean_votes += 1
    # ISC vote added after we compute it below

    scores['_stress_votes'] = stress_votes
    scores['_clean_votes'] = clean_votes
    scores['_n_voters'] = n_voters

    return scores


# ─────────────────────────────────────────────────────────────────────
# Price-based: ISC variance EWS and lookforward returns
# ─────────────────────────────────────────────────────────────────────
def compute_isc_at_observation(price_series, obs_date):
    """Compute variance EWS regime using only prices through obs_date."""
    if price_series is None:
        return None, None

    # Slice to prices on or before obs_date
    pre_obs = price_series[price_series.index <= pd.Timestamp(obs_date)]
    if len(pre_obs) < 90:
        return None, None

    try:
        result = compute_variance_score(
            pre_obs,
            window_days=VARIANCE_WINDOW_DAYS,
            rolling_window=VARIANCE_ROLLING_WINDOW,
        )
        if result and 'error' not in result:
            return result.get('regime'), result.get('mean_variance')
    except Exception as e:
        log.debug(f'variance_score failed: {e}')
    return None, None


def compute_lookforward_outcomes(price_series, obs_date, horizons_q=LOOKFORWARD_HORIZONS_Q):
    """For each lookforward horizon, compute the realized log return and
    max drawdown from observation date forward."""
    outcomes = {}
    if price_series is None:
        for h in horizons_q:
            outcomes[f'logret_{h}q'] = None
            outcomes[f'maxdd_{h}q'] = None
        return outcomes

    # Price at observation
    pre_obs = price_series[price_series.index <= pd.Timestamp(obs_date)]
    if len(pre_obs) == 0:
        for h in horizons_q:
            outcomes[f'logret_{h}q'] = None
            outcomes[f'maxdd_{h}q'] = None
        return outcomes
    price_at_obs = float(pre_obs.iloc[-1])

    for h in horizons_q:
        target_date = pd.Timestamp(obs_date) + pd.Timedelta(days=h * 91)  # ~91 days/quarter
        forward = price_series[(price_series.index > pd.Timestamp(obs_date)) &
                               (price_series.index <= target_date)]
        if len(forward) < 10:  # need at least ~2 weeks of data
            outcomes[f'logret_{h}q'] = None
            outcomes[f'maxdd_{h}q'] = None
            continue

        price_at_horizon = float(forward.iloc[-1])
        logret = math.log(price_at_horizon / price_at_obs) if price_at_obs > 0 else None

        # Max drawdown over the lookforward window
        peak = price_at_obs
        max_dd = 0.0
        for p in forward.values:
            if p > peak:
                peak = p
            if peak > 0:
                dd = (p - peak) / peak  # negative number
                if dd < max_dd:
                    max_dd = dd

        outcomes[f'logret_{h}q'] = round(logret, 4) if logret is not None else None
        outcomes[f'maxdd_{h}q'] = round(max_dd, 4)

    return outcomes


# ─────────────────────────────────────────────────────────────────────
# Per-ticker pipeline
# ─────────────────────────────────────────────────────────────────────
def backtest_ticker(ticker, cik, sector=''):
    """Run the full backtest pipeline for one ticker.
    Returns dict ready to write to CSV, or None if data insufficient."""
    log.info(f'  → {ticker} (CIK {cik})')

    # 1. EDGAR facts
    facts = fetch_company_facts(cik)
    if facts is None:
        return {'ticker': ticker, 'status': 'edgar_fetch_failed'}

    # 2. Extract sliced series
    raw = build_historical_raw(facts, obs_q=OBS_QUARTER)
    if raw is None:
        return {'ticker': ticker, 'status': 'edgar_extract_failed'}

    # Check we have enough quarters
    rev = raw.get('revenue')
    if rev is None or len(rev) < OBS_QUARTER:
        return {'ticker': ticker, 'status': f'insufficient_history_{len(rev) if rev is not None else 0}q'}

    # 3. Observation date = end of OBS_QUARTER
    obs_date = compute_observation_date(raw, obs_q=OBS_QUARTER)
    if obs_date is None:
        return {'ticker': ticker, 'status': 'no_obs_date'}

    # 4. Compute fundamental scores at observation
    scores = compute_scores_at_observation(raw, obs_q=OBS_QUARTER)

    # 5. Pull full daily price history (5 years gives us 8Q lookback + 12Q forward)
    full_prices, _ = fetch_daily_prices(ticker, days=1825)
    if full_prices is None or len(full_prices) < 90:
        return {'ticker': ticker, 'status': 'no_price_data', 'obs_date': str(obs_date), **scores}

    # 6. ISC variance EWS at observation
    isc_regime, isc_variance = compute_isc_at_observation(full_prices, obs_date)
    scores['isc_regime'] = isc_regime
    scores['isc_variance'] = isc_variance

    # 7. Finalize composite with ISC vote
    if isc_regime:
        scores['_n_voters'] += 1
        if isc_regime == 'distressed': scores['_stress_votes'] += 1
        elif isc_regime == 'stable': scores['_clean_votes'] += 1

    if scores['_n_voters'] >= 2:
        if scores['_stress_votes'] >= 2: scores['composite_class'] = 'high_risk'
        elif scores['_clean_votes'] >= 2: scores['composite_class'] = 'low_risk'
        else: scores['composite_class'] = 'mixed'

    # 8. Lookforward outcomes
    outcomes = compute_lookforward_outcomes(full_prices, obs_date)

    # 9. Assemble row
    out = {
        'ticker': ticker,
        'cik': cik,
        'sector': sector,
        'obs_date': str(obs_date),
        'status': 'ok',
        'n_quarters_available': len(rev),
        'isc_regime': scores.get('isc_regime'),
        'isc_variance': scores.get('isc_variance'),
        'altman_z': scores.get('altman_z'),
        'altman_class': scores.get('altman_class'),
        'piotroski_f': scores.get('piotroski_f'),
        'piotroski_class': scores.get('piotroski_class'),
        'beneish_m': scores.get('beneish_m'),
        'beneish_class': scores.get('beneish_class'),
        'composite_class': scores.get('composite_class'),
        'composite_stress_votes': scores.get('_stress_votes'),
        'composite_clean_votes': scores.get('_clean_votes'),
        'composite_n_voters': scores.get('_n_voters'),
        **outcomes,
    }
    return out


# ─────────────────────────────────────────────────────────────────────
# Aggregation / horse race
# ─────────────────────────────────────────────────────────────────────
def compute_auc(scores, labels):
    """ROC AUC. scores: float, higher = more positive class."""
    if not scores or len(scores) != len(labels):
        return None
    pairs = sorted(zip(scores, labels))
    n_pos = sum(1 for s, l in pairs if l == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Tied-rank Mann-Whitney
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j

    sum_pos_ranks = sum(ranks[idx] for idx, (_, l) in enumerate(pairs) if l == 1)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def horse_race(results, horizons=LOOKFORWARD_HORIZONS_Q):
    """For each (scoring system, horizon) compute AUC at discriminating
    'large drawdown' (maxdd < -0.15)."""
    valid = [r for r in results if r.get('status') == 'ok']
    if not valid:
        return []

    races = []
    systems = [
        ('ISC variance EWS', 'isc_regime', {'distressed'}, {'stable'}),
        ('Altman Z',        'altman_class', {'distressed'}, {'safe'}),
        ('Piotroski F',     'piotroski_class', {'weak'}, {'strong'}),
        ('Beneish M',       'beneish_class', {'flagged','watch'}, {'clean'}),
        ('Composite',       'composite_class', {'high_risk'}, {'low_risk'}),
    ]

    for sys_name, key, stress_set, clean_set in systems:
        for h in horizons:
            outcome_key = f'maxdd_{h}q'
            rows = [r for r in valid
                    if r.get(key) is not None and r.get(outcome_key) is not None]
            if len(rows) < 10:
                continue

            # AUC: positive class = "large drawdown" (maxdd < -0.15)
            scores = [1.0 if r[key] in stress_set else 0.0 for r in rows]
            labels = [1 if r[outcome_key] < -0.15 else 0 for r in rows]
            auc = compute_auc(scores, labels)

            # Mean outcome by stress vs clean classification
            stress_outcomes = [r[outcome_key] for r in rows if r[key] in stress_set]
            clean_outcomes = [r[outcome_key] for r in rows if r[key] in clean_set]
            mean_stress = sum(stress_outcomes)/len(stress_outcomes) if stress_outcomes else None
            mean_clean  = sum(clean_outcomes)/len(clean_outcomes) if clean_outcomes else None

            races.append({
                'system': sys_name,
                'horizon_q': h,
                'n': len(rows),
                'auc': auc,
                'n_stress': len(stress_outcomes),
                'mean_maxdd_stress': mean_stress,
                'n_clean': len(clean_outcomes),
                'mean_maxdd_clean': mean_clean,
                'separation': (mean_clean - mean_stress) if (mean_stress is not None and mean_clean is not None) else None,
            })
    return races


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Walk-forward backtest pipeline')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only run first N tickers (smoke test). Default = full snapshot.')
    ap.add_argument('--tickers', nargs='+', default=None,
                    help='Specific tickers to backtest (overrides --limit)')
    ap.add_argument('--snapshot', default=str(SNAPSHOT_CSV),
                    help='Path to snapshot CSV with ticker + CIK columns')
    args = ap.parse_args()

    # Load ticker list from snapshot
    log.info(f'Loading tickers from {args.snapshot}')
    df = pd.read_csv(args.snapshot)

    if args.tickers:
        df = df[df['ticker'].isin([t.upper() for t in args.tickers])]
        log.info(f'Filtered to {len(df)} specified tickers')
    elif args.limit:
        df = df.head(args.limit)
        log.info(f'Limited to first {len(df)} tickers (smoke test)')

    log.info(f'Running backtest on {len(df)} tickers...')
    log.info(f'  Observation point: end of quarter {OBS_QUARTER}')
    log.info(f'  Lookforward horizons: {LOOKFORWARD_HORIZONS_Q} quarters')

    results = []
    t0 = time.time()
    for i, row in df.iterrows():
        ticker = row['ticker']
        cik = int(row['cik']) if pd.notna(row.get('cik')) else None
        sector = row.get('sector', '')
        if cik is None:
            log.warning(f'  skip {ticker}: no CIK')
            continue

        try:
            result = backtest_ticker(ticker, cik, sector)
            if result:
                results.append(result)
        except Exception as e:
            log.error(f'  {ticker} failed: {e}')
            log.debug(traceback.format_exc())
            results.append({'ticker': ticker, 'status': f'error:{type(e).__name__}'})

        # Polite rate limit (SEC EDGAR is OK with 10/sec but we add a buffer)
        time.sleep(0.15)

        if (i+1) % 25 == 0:
            elapsed = time.time() - t0
            log.info(f'  progress: {i+1}/{len(df)} done in {elapsed:.0f}s ({elapsed/(i+1):.1f}s/ticker)')

    elapsed = time.time() - t0
    log.info(f'\nBacktest complete: {len(results)} tickers in {elapsed:.0f}s')

    # Write per-ticker results
    obs_csv = RESULTS_DIR / 'backtest_observations.csv'
    fieldnames = [
        'ticker', 'cik', 'sector', 'obs_date', 'status', 'n_quarters_available',
        'isc_regime', 'isc_variance',
        'altman_z', 'altman_class',
        'piotroski_f', 'piotroski_class',
        'beneish_m', 'beneish_class',
        'composite_class', 'composite_stress_votes', 'composite_clean_votes', 'composite_n_voters',
    ] + [f'logret_{h}q' for h in LOOKFORWARD_HORIZONS_Q] + [f'maxdd_{h}q' for h in LOOKFORWARD_HORIZONS_Q]

    with obs_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in results:
            w.writerow(r)
    log.info(f'Wrote {obs_csv}')

    # Horse race summary
    races = horse_race(results)
    summary_path = RESULTS_DIR / 'backtest_summary.txt'
    with summary_path.open('w') as f:
        f.write('='*78 + '\n')
        f.write(f'THE FILTER LAB — WALK-FORWARD BACKTEST SUMMARY\n')
        f.write(f'Generated: {datetime.now().isoformat()}\n')
        f.write(f'Tickers attempted: {len(df)}\n')
        f.write(f'Successful observations: {sum(1 for r in results if r.get("status") == "ok")}\n')
        f.write(f'Observation point: end of quarter {OBS_QUARTER}\n')
        f.write('='*78 + '\n\n')

        f.write('Horse race — AUC for "regime predicts maxdd < -15%" by horizon\n')
        f.write('-'*78 + '\n')
        f.write(f'{"System":<22} {"Horizon":>8} {"N":>5} {"AUC":>6} {"MeanDD_stress":>14} {"MeanDD_clean":>13} {"Sep":>6}\n')
        f.write('-'*78 + '\n')
        for r in races:
            auc = f'{r["auc"]:.3f}' if r['auc'] is not None else '   --'
            ms  = f'{r["mean_maxdd_stress"]:>+.1%}' if r['mean_maxdd_stress'] is not None else '       --'
            mc  = f'{r["mean_maxdd_clean"]:>+.1%}'  if r['mean_maxdd_clean']  is not None else '       --'
            sp  = f'{r["separation"]:>+.1%}' if r['separation'] is not None else '    --'
            f.write(f'{r["system"]:<22} {r["horizon_q"]:>4}Q   {r["n"]:>5} {auc:>6} {ms:>14} {mc:>13} {sp:>6}\n')

        f.write('\n')
        f.write('Interpretation:\n')
        f.write('  - AUC > 0.65: strong discriminative power\n')
        f.write('  - AUC 0.55-0.65: modest discriminative power\n')
        f.write('  - AUC ≈ 0.50: chance (no signal)\n')
        f.write('  - Sep column: how much MORE drawdown stress-class shows vs clean-class.\n')
        f.write('    Negative sep = stress class shows BIGGER drawdown (signal in expected direction).\n')

    log.info(f'Wrote {summary_path}')

    # Print summary to stdout too
    print('\n' + summary_path.read_text())

    # ISC distressed cohort detail
    distressed = [r for r in results if r.get('isc_regime') == 'distressed' and r.get('status') == 'ok']
    if distressed:
        dpath = RESULTS_DIR / 'backtest_distressed_cohort.csv'
        with dpath.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            w.writeheader()
            for r in distressed:
                w.writerow(r)
        log.info(f'Wrote {dpath} ({len(distressed)} tickers ISC-distressed at observation)')


if __name__ == '__main__':
    main()
