"""
ISC Financial — Variance EWS Score
====================================
Implements the paper-validated variance signal for live financial analysis.

Reference: Malone (2026)
  - Filter Collapse paper (P07): Zenodo DOI 10.5281/zenodo.18940081
    Variance-only AUC = 0.86-0.88 across 20-90 day windows
  - IRFA submission: AUC = 0.963 [0.935, 0.985] for mean rolling variance
                     AUC = 0.796 [0.722, 0.867] for variance trend (Spearman rho)

Methodology:
  1. Daily log returns from adjusted close prices
  2. Rolling 90-day variance (paper's primary parameterization)
  3. Mean variance + Spearman trend over the analysis window
  4. Regime classification considers BOTH magnitude and trend
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, no scipy dependency."""
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
    return float(1 - 6 * np.dot(d, d) / (n * (n*n - 1)))


def classify_regime(mean_var: float, variance_trend: float, var_ratio: float) -> str:
    """
    Two-dimensional regime classifier considering magnitude and trend.
    
    Calibrated against paper distributions:
      - Stable controls: mean_var 0.04-0.10, trend negative or near-zero, ratio ~0.5-1.0
      - Pre-event distress: mean_var 0.10-0.50+, trend positive (often >0.5), ratio >1.5
    
    Returns one of:
      - 'stable':   low magnitude AND not rising
      - 'rising':   trajectory clearly upward, magnitude moderate
      - 'elevated': magnitude high but not strongly rising
      - 'distressed': magnitude high AND rising (paper's strongest signal),
                      OR variance very high regardless of trend (Batch 7h.10).
    """
    # Treat NaN trend as zero for classification
    if variance_trend is None or np.isnan(variance_trend):
        variance_trend = 0.0
    if var_ratio is None or np.isnan(var_ratio):
        var_ratio = 1.0

    # Batch 7h.10: high-magnitude override.
    # When variance is extremely high (>= 0.5), classify as distressed regardless
    # of trend direction. Rationale: variance >= 0.5 represents annualized vol of
    # ~70%+ — extreme dispersion that historically only appears in genuinely
    # distressed firms (Spirit FY22, BBBY pre-bankruptcy, late-stage Lehman).
    # Trend can be flat or even negative if the variance has already peaked,
    # but the absolute level is the dominant signal at that magnitude.
    if mean_var >= 0.5:
        return 'distressed'

    # Strong rising signal regardless of current magnitude
    rising_strong = variance_trend > 0.5 and var_ratio > 2.0

    if mean_var < 0.10:
        if rising_strong:
            return 'rising'
        return 'stable'
    elif mean_var < 0.25:
        if rising_strong or variance_trend > 0.4:
            return 'rising'
        return 'elevated'
    else:  # mean_var >= 0.25
        if variance_trend > 0.3:
            return 'distressed'
        return 'elevated'


def detect_data_anomaly(prices: pd.Series) -> Optional[str]:
    """
    Detect data quality issues that would invalidate the variance score.
    Returns reason string if anomaly detected, None if data looks clean.
    """
    if prices is None or len(prices) == 0:
        return 'empty_series'
    
    # Check for duplicate dates (e.g., post-delisting OTC pink-sheet trading)
    if hasattr(prices.index, 'duplicated'):
        n_dup = prices.index.duplicated().sum()
        if n_dup > len(prices) * 0.05:
            return f'duplicate_dates ({n_dup} duplicates)'
    
    # Check for zero or negative prices
    n_invalid = ((prices <= 0) | prices.isna()).sum()
    if n_invalid > len(prices) * 0.10:
        return f'invalid_prices ({n_invalid} zero/negative/NaN)'
    
    # Check for extreme single-day moves (likely data errors)
    clean = prices.dropna()
    clean = clean[clean > 0]
    if len(clean) >= 2:
        log_rets = np.log(clean / clean.shift(1)).dropna()
        # >50% single-day move is almost always a data error in modern markets
        n_extreme = (log_rets.abs() > 0.5).sum()
        if n_extreme > 5:
            return f'extreme_returns ({n_extreme} >50% single-day moves)'
    
    return None


def compute_variance_score(
    prices: pd.Series,
    window_days: int = 252,
    rolling_window: int = 90,
) -> Optional[Dict[str, Any]]:
    """
    Compute variance EWS score from a daily price series.
    
    Parameters
    ----------
    prices : pd.Series
        Daily adjusted close prices, indexed by date.
    window_days : int
        Total analysis window in trading days (default 252 = 1 year).
    rolling_window : int
        Rolling variance window in trading days (default 90, paper primary).
    
    Returns
    -------
    dict or None
        Variance score with regime classification, or None if insufficient/anomalous data.
    """
    if prices is None or len(prices) == 0:
        return None
    
    # Drop duplicates (keep latest entry for each date) before validation
    if hasattr(prices.index, 'duplicated'):
        prices = prices[~prices.index.duplicated(keep='last')]
    
    # Anomaly detection
    anomaly = detect_data_anomaly(prices)
    if anomaly:
        return {'error': f'data_anomaly: {anomaly}', 'regime': 'insufficient_data'}
    
    prices = prices.dropna()
    prices = prices[prices > 0]
    
    if len(prices) < rolling_window + 30:
        return None
    
    # Restrict to window_days most recent observations
    if len(prices) > window_days:
        prices = prices.iloc[-window_days:]
    
    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < rolling_window + 10:
        return None
    
    # Rolling variance, annualized
    rolling_var_daily = log_returns.rolling(window=rolling_window, min_periods=rolling_window // 2).var()
    rolling_var_annualized = rolling_var_daily * 252
    valid_var = rolling_var_annualized.dropna()
    
    if len(valid_var) < 10:
        return None
    
    mean_var = float(valid_var.mean())
    latest_var = float(valid_var.iloc[-1])
    baseline_var = float(valid_var.iloc[:max(1, len(valid_var)//4)].mean())
    var_ratio = latest_var / baseline_var if baseline_var > 0 else float('nan')
    
    time_index = np.arange(len(valid_var))
    var_trend = spearman_rho(time_index, valid_var.values)
    
    regime = classify_regime(mean_var, var_trend, var_ratio)
    
    return {
        'mean_variance': round(mean_var, 6),
        'variance_trend': round(var_trend, 4) if not np.isnan(var_trend) else None,
        'latest_variance': round(latest_var, 6),
        'baseline_variance': round(baseline_var, 6),
        'variance_ratio': round(var_ratio, 3) if not np.isnan(var_ratio) else None,
        'regime': regime,
        'n_obs': len(valid_var),
        'window_days': window_days,
        'rolling_window': rolling_window,
        'data_quality': 'clean',
    }
