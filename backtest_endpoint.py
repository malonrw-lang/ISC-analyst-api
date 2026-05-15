"""
backtest_endpoint.py — v2 (rolling obs_q support)
==================================================
Adds a /backtest/{ticker} endpoint to the ISC Analyst+ API.

CHANGES from v1:
  - Data store is now NESTED: _BACKTEST_DATA[ticker][obs_date] = row.
    Previously was flat (one row per ticker).
  - /backtest/{ticker} accepts ?obs_date=YYYY-MM-DD query param to pick a
    specific observation date. If omitted, returns the LATEST available
    obs_date for the ticker (backwards compatible with existing frontend).
  - NEW endpoint: /backtest/{ticker}/obs_dates returns list of all available
    obs_dates for a ticker, plus their obs_q values.
  - Response now includes an obs_date_source field: 'latest_available' (default)
    or 'user_supplied' (when obs_date query param was passed).

Loads backtest_per_stock.csv into memory at import time, then serves
historical multi-obs observations + forward outcomes as structured JSON.

Author: Ryan W. Malone
"""
import os
import csv
import math
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse


# ──────────────────────────────────────────────────────────────────────────────
# In-memory backtest data store
# ──────────────────────────────────────────────────────────────────────────────

# NESTED: _BACKTEST_DATA[ticker_upper][obs_date_str] = row_dict
# Previously was flat. This change lets us hold multiple obs_dates per ticker.
_BACKTEST_DATA: Dict[str, Dict[str, Dict[str, Any]]] = {}
_LOAD_STATUS: Dict[str, Any] = {
    'loaded': False,
    'error': None,
    'n_tickers': 0,
    'n_observations': 0,
    'csv_path': None,
}


def _coerce_value(key: str, raw: str):
    """Convert a CSV string cell to the right Python type for JSON output.

    Returns None for empty strings or 'nan'.
    Numeric columns get float-coerced. Regime/string columns stay as strings.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s == '' or s.lower() == 'nan':
        return None

    # String fields stay as strings
    string_fields = {
        'ticker', 'sector_bucket', 'sector', 'obs_date',
        'isc_regime', 'altman_regime', 'piotroski_regime',
        'beneish_regime', 'composite_regime',
        'closest_aligned_framework_8q',
        'outcome_regime_1q', 'outcome_regime_2q', 'outcome_regime_4q',
        'outcome_regime_8q', 'outcome_regime_12q',
        'outcome_regime',  # singular form from rolling-obs pipeline
        'price_source',
    }
    if key in string_fields:
        return s

    # Everything else: try float (or int for obs_q)
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except ValueError:
        return None


def load_backtest_data(csv_path: Optional[str] = None):
    """Load the per-stock CSV into the nested module-level dict.

    Idempotent: safe to call multiple times. Errors are caught and stored in
    _LOAD_STATUS so the endpoint can report them rather than crashing the app.

    Handles BOTH CSV formats:
      - Old (v1, one row per ticker): one obs_date per ticker, columns named
        outcome_regime_1q/2q/4q/8q/12q etc.
      - New (v3, rolling obs_q with one row per (ticker, obs_q, horizon)):
        multiple obs_dates per ticker, with obs_q and horizon_q columns.
    """
    global _BACKTEST_DATA, _LOAD_STATUS

    if csv_path is None:
        candidates = [
            Path('backtest_per_stock.csv'),
            Path('backtest_results.csv'),  # v3 pipeline default output
            Path(__file__).parent / 'backtest_per_stock.csv',
            Path(__file__).parent / 'backtest_results.csv',
            Path(os.environ.get('BACKTEST_CSV_PATH', '')),
        ]
        for c in candidates:
            if c and c.is_file():
                csv_path = str(c)
                break

    _LOAD_STATUS['csv_path'] = csv_path

    if not csv_path or not Path(csv_path).is_file():
        _LOAD_STATUS['loaded'] = False
        _LOAD_STATUS['error'] = (
            f'backtest CSV not found '
            f'(checked: backtest_per_stock.csv, backtest_results.csv, '
            f'alongside main.py, $BACKTEST_CSV_PATH)'
        )
        return

    try:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Detect format: v3 has horizon_q column, v1 doesn't
        is_long_format = bool(rows) and 'horizon_q' in rows[0]

        nested: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if is_long_format:
            # v3 format: one row per (ticker, obs_q, horizon_q).
            # Group rows by (ticker, obs_date) and merge horizon-specific
            # fields into outcome blocks.
            from collections import defaultdict
            grouped = defaultdict(list)
            for row in rows:
                t = row.get('ticker', '').strip().upper()
                d = row.get('obs_date', '').strip()
                if not t or not d:
                    continue
                grouped[(t, d)].append(row)

            for (t, d), group_rows in grouped.items():
                # Pick the first row's flat fields (ticker, obs_date, regimes
                # and scores are the same across all horizons within a group)
                base = group_rows[0]
                merged = {k: _coerce_value(k, v) for k, v in base.items()}

                # Build the "horizons" sub-structure by pulling each row's
                # horizon-specific fields under its h-key
                horizon_block = {}
                for r in group_rows:
                    h = r.get('horizon_q', '').strip()
                    if not h:
                        continue
                    horizon_block[f'{h}q'] = {
                        'total_return':           _coerce_value('total_return', r.get('total_return')),
                        'ann_return':             _coerce_value('ann_return', r.get('ann_return')),
                        'max_drawdown':           _coerce_value('max_drawdown', r.get('max_drawdown')),
                        'realized_vol':           _coerce_value('realized_vol', r.get('realized_vol')),
                        'sector_relative_return': _coerce_value('sector_relative_return',
                                                                 r.get('sector_relative_return')),
                        'outcome_regime':         _coerce_value('outcome_regime', r.get('outcome_regime')),
                        'n_forward_days':         _coerce_value('n_forward_days', r.get('n_forward_days')),
                    }
                merged['_horizon_block'] = horizon_block
                # Convert obs_q to int for cleaner downstream use
                if 'obs_q' in merged and merged['obs_q'] is not None:
                    merged['obs_q'] = int(merged['obs_q'])

                if t not in nested:
                    nested[t] = {}
                nested[t][d] = merged
        else:
            # v1 format: one row per ticker, single obs_date.
            for row in rows:
                t = row.get('ticker', '').strip().upper()
                d = row.get('obs_date', '').strip()
                if not t:
                    continue
                clean_row = {k: _coerce_value(k, v) for k, v in row.items()}
                if t not in nested:
                    nested[t] = {}
                # Use the row's obs_date as the inner key. If missing, use a
                # sentinel so it can still be addressed.
                key = d if d else '_unknown'
                nested[t][key] = clean_row

        _BACKTEST_DATA = nested
        _LOAD_STATUS['loaded'] = True
        _LOAD_STATUS['error'] = None
        _LOAD_STATUS['n_tickers'] = len(nested)
        _LOAD_STATUS['n_observations'] = sum(len(obs) for obs in nested.values())
        _LOAD_STATUS['format'] = 'rolling_obs_q' if is_long_format else 'single_obs'
    except Exception as e:
        _LOAD_STATUS['loaded'] = False
        _LOAD_STATUS['error'] = f'Failed to load CSV: {type(e).__name__}: {e}'


# ──────────────────────────────────────────────────────────────────────────────
# Response shaping
# ──────────────────────────────────────────────────────────────────────────────

def _quarter_labels(obs_date_str: str):
    """Generate calendar-quarter labels (e.g., 'Q2 2022') for Q+1 through Q+12
    given the observation date."""
    from datetime import datetime, timedelta
    labels = []
    try:
        obs = datetime.strptime(obs_date_str, '%Y-%m-%d')
    except (TypeError, ValueError):
        return [None] * 12
    for n in range(1, 13):
        midpoint = obs + timedelta(days=(n - 1) * 91 + 45)
        cal_q = (midpoint.month - 1) // 3 + 1
        yy = midpoint.year % 100
        labels.append(f"Q{cal_q} '{yy:02d}")
    return labels


def _shape_response(row: Dict[str, Any], obs_date_source: str = 'latest_available') -> Dict[str, Any]:
    """
    Reshape the row into nested JSON the frontend can render directly.
    Handles both v1 (flat, with outcome_regime_Nq fields) and v3 (with _horizon_block).
    """
    # Build outcomes block — works for both formats
    if '_horizon_block' in row:
        # v3 format: outcomes are already nested
        outcomes = row['_horizon_block']
    else:
        # v1 format: outcomes are flat fields
        def horizon_block(h):
            return {
                'total_return':           row.get(f'total_return_{h}q'),
                'sector_relative_return': row.get(f'sector_relative_return_{h}q'),
                'max_drawdown':           row.get(f'max_drawdown_{h}q'),
                'outcome_regime':         row.get(f'outcome_regime_{h}q'),
            }
        outcomes = {
            '1q':  horizon_block(1),
            '2q':  horizon_block(2),
            '4q':  horizon_block(4),
            '8q':  horizon_block(8),
            '12q': horizon_block(12),
        }

    response = {
        'ticker':           row.get('ticker'),
        'sector':           row.get('sector'),
        'sector_bucket':    row.get('sector_bucket'),
        'obs_date':         row.get('obs_date'),
        'obs_q':            row.get('obs_q'),
        'obs_date_source':  obs_date_source,
        'observation': {
            'isc': {
                'regime': row.get('isc_regime'),
                'score':  row.get('isc_score'),
                'trend':  row.get('isc_trend'),
                'ratio':  row.get('isc_ratio'),
            },
            'altman': {
                'regime': row.get('altman_regime'),
                'z':      row.get('altman_z'),
            },
            'piotroski': {
                'regime':  row.get('piotroski_regime'),
                'f_score': row.get('piotroski_f'),
            },
            'beneish': {
                'regime':  row.get('beneish_regime'),
                'm_score': row.get('beneish_m'),
            },
            'composite': {
                'regime': row.get('composite_regime'),
            },
        },
        'outcomes': outcomes,
        'quarterly_returns': [row.get(f'q{q}_return') for q in range(1, 13)],
        'quarter_labels': _quarter_labels(row.get('obs_date') or ''),
        'closest_aligned_framework_8q': row.get('closest_aligned_framework_8q'),
    }
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Route attachment
# ──────────────────────────────────────────────────────────────────────────────

def attach_backtest_routes(app):
    """Add /backtest/{ticker}, /backtest/{ticker}/obs_dates, /backtest_status,
    and /backtest_universe routes."""

    if not _LOAD_STATUS['loaded']:
        load_backtest_data()

    @app.get('/backtest/{ticker}')
    async def backtest_for_ticker(ticker: str,
                                  obs_date: Optional[str] = Query(
                                      None,
                                      description=(
                                          "Specific observation date "
                                          "YYYY-MM-DD. If omitted, returns "
                                          "the latest available obs_date "
                                          "for this ticker."
                                      ))):
        """Return historical backtest entry for a ticker.

        Behavior:
        - No obs_date param → returns the LATEST available obs_date (backwards
          compatible).
        - With obs_date param → returns that specific obs_date if available,
          else 404 with the list of available dates.

        Error codes:
        - 404 if ticker not in backtest universe.
        - 404 if ticker has no obs_date matching the supplied value.
        - 503 if backtest data isn't loaded.
        """
        if not _LOAD_STATUS['loaded']:
            raise HTTPException(
                status_code=503,
                detail={
                    'error': 'backtest_data_unavailable',
                    'message': _LOAD_STATUS.get('error') or 'CSV not loaded',
                }
            )

        t = ticker.strip().upper()
        ticker_obs = _BACKTEST_DATA.get(t)
        if ticker_obs is None or not ticker_obs:
            raise HTTPException(
                status_code=404,
                detail={
                    'error': 'ticker_not_in_backtest',
                    'ticker': t,
                    'message': (
                        f'{t} is not in the backtest universe '
                        f'({_LOAD_STATUS["n_tickers"]} tickers, '
                        f'{_LOAD_STATUS["n_observations"]} observations).'
                    ),
                }
            )

        # Pick which observation to return
        if obs_date is not None:
            requested = obs_date.strip()
            if requested not in ticker_obs:
                available = sorted(ticker_obs.keys())
                raise HTTPException(
                    status_code=404,
                    detail={
                        'error': 'obs_date_not_found',
                        'ticker': t,
                        'requested_obs_date': requested,
                        'available_obs_dates': available,
                        'message': (
                            f'{t} has no observation at {requested}. '
                            f'Available: {available}'
                        ),
                    }
                )
            row = ticker_obs[requested]
            source = 'user_supplied'
        else:
            # Default: latest available obs_date (by string sort, which works
            # for YYYY-MM-DD format)
            latest_obs_date = max(ticker_obs.keys())
            row = ticker_obs[latest_obs_date]
            source = 'latest_available'

        return JSONResponse(content=_shape_response(row, obs_date_source=source))

    @app.get('/backtest/{ticker}/obs_dates')
    async def backtest_obs_dates(ticker: str):
        """List all available obs_dates for a ticker, with their obs_q values.

        Useful for the analysis script: hit this once to discover what obs_dates
        are available, then hit /backtest/{ticker}?obs_date=... for each.
        """
        if not _LOAD_STATUS['loaded']:
            raise HTTPException(
                status_code=503,
                detail={'error': 'backtest_data_unavailable',
                        'message': _LOAD_STATUS.get('error')}
            )

        t = ticker.strip().upper()
        ticker_obs = _BACKTEST_DATA.get(t)
        if ticker_obs is None or not ticker_obs:
            raise HTTPException(
                status_code=404,
                detail={
                    'error': 'ticker_not_in_backtest',
                    'ticker': t,
                }
            )

        obs_list = []
        for d in sorted(ticker_obs.keys()):
            row = ticker_obs[d]
            obs_list.append({
                'obs_date': d,
                'obs_q': row.get('obs_q'),
            })
        return {
            'ticker': t,
            'n_observations': len(obs_list),
            'observations': obs_list,
        }

    @app.get('/backtest_status')
    async def backtest_status():
        """Diagnostic: is the backtest CSV loaded and what does it cover."""
        return {
            'loaded':         _LOAD_STATUS['loaded'],
            'error':          _LOAD_STATUS['error'],
            'n_tickers':      _LOAD_STATUS['n_tickers'],
            'n_observations': _LOAD_STATUS['n_observations'],
            'csv_path':       _LOAD_STATUS['csv_path'],
            'format':         _LOAD_STATUS.get('format', 'unknown'),
        }

    @app.get('/backtest_universe')
    async def backtest_universe():
        """Return the list of all tickers in the backtest universe.

        Useful for the frontend to know which tickers have a 'Backtest Reference'
        tab available. Sorted alphabetically.
        """
        if not _LOAD_STATUS['loaded']:
            raise HTTPException(
                status_code=503,
                detail={'error': 'backtest_data_unavailable',
                        'message': _LOAD_STATUS.get('error')}
            )
        return {
            'n_tickers':      _LOAD_STATUS['n_tickers'],
            'n_observations': _LOAD_STATUS['n_observations'],
            'tickers':        sorted(_BACKTEST_DATA.keys()),
        }
