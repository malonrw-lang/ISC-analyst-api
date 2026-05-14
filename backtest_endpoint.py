"""
backtest_endpoint.py
====================
Adds a /backtest/{ticker} endpoint to the ISC Analyst+ API.

Loads backtest_per_stock.csv into memory at import time, then serves
historical Q8 observations + forward outcomes as structured JSON.

Usage in main.py (add to the existing imports/routes section):

    from backtest_endpoint import attach_backtest_routes
    attach_backtest_routes(app)

The CSV is expected at the repo root: backtest_per_stock.csv

If the CSV is missing or fails to parse, the endpoint returns 503 with a
clear error message. The rest of the API continues to work normally.

Author: Ryan W. Malone
"""
import os
import csv
import math
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse


# ──────────────────────────────────────────────────────────────────────────────
# In-memory backtest data store
# ──────────────────────────────────────────────────────────────────────────────

# Loaded once at import-time. Module-level so multiple imports share the same
# dict and we don't re-read the CSV on every request.
_BACKTEST_DATA: Dict[str, Dict[str, Any]] = {}
_LOAD_STATUS: Dict[str, Any] = {
    'loaded': False,
    'error': None,
    'n_tickers': 0,
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
    }
    if key in string_fields:
        return s

    # Everything else: try float
    try:
        v = float(s)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except ValueError:
        return None


def load_backtest_data(csv_path: Optional[str] = None):
    """Load the per-stock CSV into the module-level dict.

    Idempotent: safe to call multiple times. Errors are caught and stored in
    _LOAD_STATUS so the endpoint can report them rather than crashing the app.
    """
    global _BACKTEST_DATA, _LOAD_STATUS

    if csv_path is None:
        # Search in common locations: cwd first, then alongside this file
        candidates = [
            Path('backtest_per_stock.csv'),
            Path(__file__).parent / 'backtest_per_stock.csv',
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
            f'backtest_per_stock.csv not found '
            f'(checked: backtest_per_stock.csv, alongside main.py, '
            f'$BACKTEST_CSV_PATH)'
        )
        return

    try:
        data = {}
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('ticker', '').strip().upper()
                if not ticker:
                    continue
                clean_row = {k: _coerce_value(k, v) for k, v in row.items()}
                data[ticker] = clean_row
        _BACKTEST_DATA = data
        _LOAD_STATUS['loaded'] = True
        _LOAD_STATUS['error'] = None
        _LOAD_STATUS['n_tickers'] = len(data)
    except Exception as e:
        _LOAD_STATUS['loaded'] = False
        _LOAD_STATUS['error'] = f'Failed to load CSV: {type(e).__name__}: {e}'


# ──────────────────────────────────────────────────────────────────────────────
# Response shaping
# ──────────────────────────────────────────────────────────────────────────────

def _quarter_labels(obs_date_str: str):
    """Generate calendar-quarter labels (e.g., 'Q2 2022') for Q+1 through Q+12
    given the Q8 observation date.

    Each Q+N covers approximately the calendar quarter starting ~91 days after
    obs_date * (N-1). We use the observation date as the anchor.
    """
    from datetime import datetime, timedelta
    labels = []
    try:
        obs = datetime.strptime(obs_date_str, '%Y-%m-%d')
    except (TypeError, ValueError):
        return [None] * 12
    for n in range(1, 13):
        # Midpoint of the Nth forward quarter
        midpoint = obs + timedelta(days=(n - 1) * 91 + 45)
        cal_q = (midpoint.month - 1) // 3 + 1
        yy = midpoint.year % 100
        labels.append(f"Q{cal_q} '{yy:02d}")
    return labels


def _shape_response(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reshape the flat CSV row into nested JSON the frontend can render directly.

    Output structure:
      {
        ticker, sector, sector_bucket, obs_date,
        observation: {            # what the frameworks said at Q8
          isc: { regime, score, trend, ratio },
          altman: { regime, z },
          piotroski: { regime, f_score },
          beneish: { regime, m_score },
          composite: { regime },
        },
        outcomes: {               # what actually happened at each horizon
          '1q': { total_return, sector_relative_return, max_drawdown, outcome_regime },
          '2q': { ... },
          '4q': { ... },
          '8q': { ... },
          '12q': { ... },
        },
        quarterly_returns: [ q1, q2, ..., q12 ],
        closest_aligned_framework_8q: 'ISC (Variance EWS)' or null,
      }
    """
    def horizon_block(h):
        return {
            'total_return':           row.get(f'total_return_{h}q'),
            'sector_relative_return': row.get(f'sector_relative_return_{h}q'),
            'max_drawdown':           row.get(f'max_drawdown_{h}q'),
            'outcome_regime':         row.get(f'outcome_regime_{h}q'),
        }

    return {
        'ticker':        row.get('ticker'),
        'sector':        row.get('sector'),
        'sector_bucket': row.get('sector_bucket'),
        'obs_date':      row.get('obs_date'),
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
        'outcomes': {
            '1q':  horizon_block(1),
            '2q':  horizon_block(2),
            '4q':  horizon_block(4),
            '8q':  horizon_block(8),
            '12q': horizon_block(12),
        },
        'quarterly_returns': [row.get(f'q{q}_return') for q in range(1, 13)],
        'quarter_labels': _quarter_labels(row.get('obs_date')),
        'closest_aligned_framework_8q': row.get('closest_aligned_framework_8q'),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Route attachment
# ──────────────────────────────────────────────────────────────────────────────

def attach_backtest_routes(app):
    """Add /backtest/{ticker}, /backtest_status, and /backtest_universe routes."""

    # Load data on first import
    if not _LOAD_STATUS['loaded']:
        load_backtest_data()

    @app.get('/backtest/{ticker}')
    async def backtest_for_ticker(ticker: str):
        """Return historical backtest entry for a ticker.

        404 if ticker not in the backtest universe.
        503 if the backtest data isn't loaded (CSV missing or parse failed).
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
        row = _BACKTEST_DATA.get(t)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail={
                    'error': 'ticker_not_in_backtest',
                    'ticker': t,
                    'message': (
                        f'{t} is not in the backtest universe. The backtest '
                        f'covered {_LOAD_STATUS["n_tickers"]} S&P 500 tickers '
                        f'with sufficient EDGAR + price history at the time '
                        f'the snapshot was generated.'
                    ),
                }
            )

        return JSONResponse(content=_shape_response(row))

    @app.get('/backtest_status')
    async def backtest_status():
        """Diagnostic: is the backtest CSV loaded and how many tickers does it cover."""
        return {
            'loaded':    _LOAD_STATUS['loaded'],
            'error':     _LOAD_STATUS['error'],
            'n_tickers': _LOAD_STATUS['n_tickers'],
            'csv_path':  _LOAD_STATUS['csv_path'],
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
            'n_tickers': _LOAD_STATUS['n_tickers'],
            'tickers':   sorted(_BACKTEST_DATA.keys()),
        }
