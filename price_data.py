"""
Daily price data fetcher with multi-source fallback.
==================================================

SINGLE PRICE BOUNDARY. The rest of the system depends ONLY on
fetch_daily_prices(ticker, days) -> (pd.Series | None, source_tag).
Analytics consume the Series and are vendor-agnostic. To add a data source:
add a _fetch_<vendor>(ticker, days) -> pd.Series | None function and insert
it into the fetch_daily_prices fallback chain. Do NOT call vendor price APIs
anywhere else in the codebase.

Sources tried in order:
  1. Tiingo (preferred) — requires TIINGO_TOKEN env var, free tier 500 req/day
  2. Stooq (fallback)   — no auth, free, CSV format

Returns a pandas Series of daily adjusted close prices indexed by date,
suitable as input to compute_variance_score().

Drops yfinance entirely — it has been unreliable on Render free tier IPs.
"""
import os
import io
import pandas as pd
import numpy as np
import requests
from typing import Optional, Tuple

def _get_tiingo_token():
    """Read token at call time, not import time (Render populates env vars late)."""
    return os.environ.get('TIINGO_TOKEN', '')
TIMEOUT_SEC = 12

UA_BROWSER = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}


def _fetch_tiingo(ticker: str, days: int = 400) -> Optional[pd.Series]:
    """
    Fetch daily adjusted close prices from Tiingo.
    Returns Series indexed by date, or None on failure.
    """
    TIINGO_TOKEN = _get_tiingo_token()
    if not TIINGO_TOKEN:
        return None
    
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=days)
    
    url = 'https://api.tiingo.com/tiingo/daily/{}/prices'.format(ticker.lower())
    params = {
        'startDate': start_date.strftime('%Y-%m-%d'),
        'endDate':   end_date.strftime('%Y-%m-%d'),
        'token':     TIINGO_TOKEN,
        'format':    'csv',
        'columns':   'date,adjClose,adjVolume',
    }

    try:
        r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
        if r.status_code != 200:
            return None

        df = pd.read_csv(io.StringIO(r.text))
        if df.empty:
            return None

        # Normalise column names to lowercase for header-casing resilience
        df.columns = [c.strip().lower() for c in df.columns]
        if 'adjclose' not in df.columns or 'date' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df[df['adjclose'] > 0]

        if len(df) < 30:
            return None

        return pd.Series(df['adjclose'].values, index=df['date'].values)
    except Exception:
        return None


def _fetch_stooq(ticker: str, days: int = 400) -> Optional[pd.Series]:
    """
    Fetch daily prices from Stooq (CSV download).
    Stooq returns full available history; we trim to the requested window.
    Returns Series indexed by date, or None on failure.
    
    Note: Stooq returns Close prices only, not split/dividend-adjusted.
    For variance computation this is acceptable because we use log RETURNS,
    which are unaffected by uniform scaling — but we should not advertise
    this as adjusted. Stooq does adjust for stock splits historically.
    """
    url = 'https://stooq.com/q/d/l/?s={}.us&i=d'.format(ticker.lower())
    
    try:
        r = requests.get(url, timeout=TIMEOUT_SEC, headers=UA_BROWSER)
        if r.status_code != 200:
            return None
        
        text = r.text
        if len(text) < 100 or 'No data' in text:
            return None
        
        df = pd.read_csv(io.StringIO(text))
        if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df[df['Close'] > 0]
        
        # Trim to requested window
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df['Date'] >= cutoff]
        
        if len(df) < 30:
            return None
        
        return pd.Series(df['Close'].values, index=df['Date'].values)
    except Exception:
        return None


def fetch_daily_prices(ticker: str, days: int = 400) -> Tuple[Optional[pd.Series], str]:
    """
    Fetch daily prices using fallback chain.
    
    Returns (price_series, source_name):
      - price_series: pd.Series indexed by date, or None on total failure
      - source_name: 'tiingo' | 'stooq' | 'failed'
    
    Parameters:
      ticker (str): US equity ticker, e.g. 'AAPL', 'MSFT'
      days (int):   how many days of history to fetch (default 730 = 2y)
    """
    ticker = ticker.upper().strip()
    
    # Try Tiingo first
    s = _fetch_tiingo(ticker, days)
    if s is not None and len(s) >= 30:
        return s, 'tiingo'
    
    # Fall back to Stooq
    s = _fetch_stooq(ticker, days)
    if s is not None and len(s) >= 30:
        return s, 'stooq'
    
    return None, 'failed'


def fetch_basic_market_metadata(ticker: str) -> dict:
    """
    Fetch lightweight market metadata (price, market cap, sector) from Tiingo.
    Returns dict with whatever fields it could populate; missing fields are None.
    
    This replaces the metadata that yfinance was returning. Note that Tiingo's
    free tier offers daily prices well, but their /tiingo/daily/{ticker} endpoint
    has limited fundamentals access — sector/industry/marketCap require their
    paid plan. We try, accept what we get, leave the rest as None.
    """
    ticker = ticker.upper().strip()
    out = {
        'price': None, 'high_52w': None, 'low_52w': None,
        'market_cap': None, 'company_name': ticker,
        'sector': None, 'industry': None, 'description': '',
    }
    TIINGO_TOKEN = _get_tiingo_token()
    if not TIINGO_TOKEN:
        return out
    
    try:
        # Tiingo metadata endpoint
        url = 'https://api.tiingo.com/tiingo/daily/{}'.format(ticker.lower())
        r = requests.get(url, params={'token': TIINGO_TOKEN}, timeout=TIMEOUT_SEC)
        if r.status_code == 200:
            data = r.json()
            out['company_name'] = data.get('name') or ticker
            out['description'] = data.get('description', '')[:300] if data.get('description') else ''
    except Exception:
        pass
    
    return out


# ── Self-test (run manually) ─────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for ticker in sys.argv[1:]:
            s, src = fetch_daily_prices(ticker)
            if s is not None:
                print(f"{ticker:<8} {src:<8} n={len(s):<5} latest={s.iloc[-1]:.2f} on {s.index[-1].date()}")
            else:
                print(f"{ticker:<8} FAILED via all sources")
    else:
        print("Usage: python price_data.py AAPL META NVDA")
        print(f"TIINGO_TOKEN set: {bool(TIINGO_TOKEN)}")
