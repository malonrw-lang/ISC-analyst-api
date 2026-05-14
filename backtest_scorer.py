"""
backtest_scorer.py
==================
Slice-aware scoring for the Filter Lab backtest.

Each scoring function takes the full EDGAR raw dict-of-Series and an integer
`obs_q` (1-indexed quarter position, where obs_q=8 means "use only the first
8 quarters of available history"). All TTM and last() operations are computed
on the slice [0:obs_q], so a Q8 call sees nothing from Q9 onward.

Validates: scorer at obs_q == len(series) should give numerically identical
results to the main.py scoring path in analyze() (modulo floating-point noise).

Author: Ryan W. Malone
Project: The Filter Lab
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Slice-aware versions of main.py's helpers.
# These differ from main.py's helpers ONLY in that they operate on a series
# slice ending at obs_q, not the full series.
# ──────────────────────────────────────────────────────────────────────────────

def slice_ttm(series, obs_q):
    """Sum of last 4 quarters within [0:obs_q] of series. None if <4 quarters."""
    if series is None or len(series) == 0 or obs_q < 4:
        return None
    s = series.iloc[:obs_q].dropna()
    if len(s) < 4:
        return None
    val = float(s.iloc[-4:].sum())
    if np.isnan(val) or np.isinf(val):
        return None
    return round(val, 3)


def slice_last(series, obs_q):
    """Most recent observation within [0:obs_q]. None if no observations."""
    if series is None or len(series) == 0 or obs_q < 1:
        return None
    s = series.iloc[:obs_q].dropna()
    if len(s) == 0:
        return None
    val = float(s.iloc[-1])
    if np.isnan(val) or np.isinf(val):
        return None
    return round(val, 3)


def slice_trend_up(series, obs_q, n=4):
    """True if value at end of slice > value n periods earlier."""
    if series is None or obs_q < n + 1:
        return None
    s = series.iloc[:obs_q].dropna()
    if len(s) < n + 1:
        return None
    return float(s.iloc[-1]) > float(s.iloc[-n - 1])


def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    v = a / b
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Altman Z (matches main.py's compute_altman_z signature)
# ──────────────────────────────────────────────────────────────────────────────

def altman_z_at(raw, obs_q, market_cap_m=None):
    """
    Compute Altman Z at observation quarter obs_q.

    Note on market cap: at historical observation points we don't have a
    point-in-time market cap. We fall back to TA*1.2 if not provided
    (matches main.py's default for the same reason).
    """
    ta = slice_last(raw.get('total_assets'), obs_q)
    re = slice_last(raw.get('retained_earnings'), obs_q)
    rev_ttm = slice_ttm(raw.get('revenue'), obs_q)
    oi_ttm = slice_ttm(raw.get('operating_income'), obs_q)
    tl = slice_last(raw.get('total_liabilities'), obs_q)
    if tl is None and ta is not None:
        te = slice_last(raw.get('total_equity'), obs_q)
        if te is not None:
            tl = round(ta - te, 2)
    ca = slice_last(raw.get('current_assets'), obs_q)
    cl = slice_last(raw.get('current_liabilities'), obs_q)

    try:
        if not ta or ta == 0 or not tl or tl == 0:
            return None
        wc = (ca or 0) - (cl or 0)
        X1 = wc / ta
        X2 = (re or 0) / ta
        X3 = (oi_ttm or 0) / ta
        X4 = (market_cap_m or ta * 1.2) / tl
        X5 = (rev_ttm or 0) / ta
        z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
        if np.isnan(z) or np.isinf(z):
            return None
        return round(z, 3)
    except Exception:
        return None


def altman_regime(z):
    """Altman's canonical zones."""
    if z is None:
        return 'unknown'
    if z < 1.81:
        return 'distress'
    if z < 3.0:
        return 'grey'
    return 'safe'


# ──────────────────────────────────────────────────────────────────────────────
# Piotroski F-Score
# ──────────────────────────────────────────────────────────────────────────────

def piotroski_at(raw, obs_q):
    """Compute Piotroski F-Score at observation quarter obs_q."""
    ni_ttm = slice_ttm(raw.get('net_income'), obs_q)
    cfo_ttm = slice_ttm(raw.get('cfo'), obs_q)
    rev_ttm = slice_ttm(raw.get('revenue'), obs_q)
    ta = slice_last(raw.get('total_assets'), obs_q)
    ca = slice_last(raw.get('current_assets'), obs_q)
    cl = slice_last(raw.get('current_liabilities'), obs_q)

    roa = safe_div(ni_ttm, ta)

    # Trend signals: comparing within [0:obs_q]
    roa_up = slice_trend_up(raw.get('net_income'), obs_q)
    leverage_up = slice_trend_up(raw.get('long_term_debt'), obs_q)
    leverage_down = (leverage_up is False) if leverage_up is not None else None
    liquidity_up = slice_trend_up(raw.get('current_assets'), obs_q)
    shares_up = slice_trend_up(raw.get('shares_outstanding'), obs_q)
    margin_up = slice_trend_up(raw.get('gross_profit'), obs_q)
    turnover_up = slice_trend_up(raw.get('revenue'), obs_q)

    signals = {
        'roa_pos':    1 if (roa or 0) > 0 else 0,
        'cfo_pos':    1 if (cfo_ttm or 0) > 0 else 0,
        'roa_up':     1 if roa_up else 0,
        'cfo_gt_ni':  1 if (cfo_ttm or 0) > (ni_ttm or 0) else 0,
        'lev_down':   1 if leverage_down else 0,
        'liq_up':     1 if liquidity_up else 0,
        'no_dilute':  1 if not shares_up else 0,
        'margin_up':  1 if margin_up else 0,
        'turnover_up':1 if turnover_up else 0,
    }
    return sum(signals.values()), signals


def piotroski_regime(f):
    """3-bucket Piotroski classification."""
    if f is None:
        return 'unknown'
    if f <= 3:
        return 'weak'
    if f <= 6:
        return 'mixed'
    return 'strong'


# ──────────────────────────────────────────────────────────────────────────────
# Beneish M-Score
# ──────────────────────────────────────────────────────────────────────────────
# For simplicity in backtest: use quarterly TTM windows rather than annual FY.
# This deviates from canonical Beneish but is consistent across the test.

def beneish_at(raw, obs_q):
    """
    Quarterly approximation of Beneish M at obs_q.

    Compares TTM ending at obs_q vs TTM ending at obs_q-4 (1 year earlier).
    Requires obs_q >= 8 (need 2 years of TTM data).

    NOT canonical Beneish — proper version uses annual FY data. This is a
    quarterly approximation used uniformly across the backtest for comparability.
    """
    if obs_q < 8:
        return None

    def _pair(key):
        """TTM at obs_q and TTM at obs_q-4. None if either missing."""
        t = slice_ttm(raw.get(key), obs_q)
        p = slice_ttm(raw.get(key), obs_q - 4)
        return (t, p)

    def _pair_stock(key):
        """Last value at obs_q and last value at obs_q-4 (balance sheet items)."""
        t = slice_last(raw.get(key), obs_q)
        p = slice_last(raw.get(key), obs_q - 4)
        return (t, p)

    rev_t, rev_p = _pair('revenue')
    recv_t, recv_p = _pair_stock('receivables')
    gp_t, gp_p = _pair('gross_profit')

    # Fallback gross profit from revenue - cost_of_revenue
    if gp_t is None and rev_t is not None:
        cor_t = slice_ttm(raw.get('cost_of_revenue'), obs_q)
        if cor_t is not None:
            gp_t = round(rev_t - cor_t, 2)
    if gp_p is None and rev_p is not None:
        cor_p = slice_ttm(raw.get('cost_of_revenue'), obs_q - 4)
        if cor_p is not None:
            gp_p = round(rev_p - cor_p, 2)

    ca_t, ca_p = _pair_stock('current_assets')
    ppe_t, ppe_p = _pair_stock('ppe_net')
    ta_t, ta_p = _pair_stock('total_assets')
    dep_t, dep_p = _pair('da')
    sga_t, sga_p = _pair('sga')
    ni_t, _ = _pair('net_income')
    cfo_t, _ = _pair('cfo')
    ltd_t, ltd_p = _pair_stock('long_term_debt')
    cl_t, cl_p = _pair_stock('current_liabilities')

    # GMI: both gross margins must be positive
    gm_t = safe_div(gp_t, rev_t) if (gp_t is not None and rev_t) else None
    gm_p = safe_div(gp_p, rev_p) if (gp_p is not None and rev_p) else None
    gmi = (gm_p / gm_t) if (gm_t and gm_p and gm_t > 0 and gm_p > 0) else None

    # DSRI
    dsri_num = safe_div(recv_t, rev_t) if (recv_t and rev_t) else None
    dsri_den = safe_div(recv_p, rev_p) if (recv_p and rev_p) else None
    dsri = safe_div(dsri_num, dsri_den)

    # AQI: other-assets share growth
    if all(v is not None for v in [ca_t, ppe_t, ta_t, ca_p, ppe_p, ta_p]) and ta_t and ta_p:
        oa_t = 1 - (ca_t + ppe_t) / ta_t
        oa_p = 1 - (ca_p + ppe_p) / ta_p
        aqi = (oa_t / oa_p) if abs(oa_p) > 0.001 else 1.0
    else:
        aqi = None

    # SGI
    sgi = safe_div(rev_t, rev_p)

    # SGAI
    sgai_num = safe_div(sga_t, rev_t) if (sga_t and rev_t) else None
    sgai_den = safe_div(sga_p, rev_p) if (sga_p and rev_p) else None
    sgai = safe_div(sgai_num, sgai_den)

    # TATA
    tata = None
    if all(v is not None for v in [ni_t, cfo_t, ta_t]) and ta_t:
        tata = (ni_t - cfo_t) / ta_t

    # LVGI
    lvgi = None
    if all(v is not None for v in [ltd_t, cl_t, ta_t, ltd_p, cl_p, ta_p]) and ta_t and ta_p:
        lev_t = (ltd_t + cl_t) / ta_t
        lev_p = (ltd_p + cl_p) / ta_p
        lvgi = safe_div(lev_t, lev_p)

    # DEPI: too sparse in quarterly data; default neutral
    depi = 1.0

    # Need at least 5 of 8 ratios
    available = [x for x in [dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi] if x is not None]
    if len(available) < 5:
        return None

    # Beneish 1999 coefficients with neutral substitution for missing
    m = (-4.84
         + 0.92 * (dsri if dsri is not None else 1.0)
         + 0.528 * (gmi if gmi is not None else 1.0)
         + 0.404 * (aqi if aqi is not None else 1.0)
         + 0.892 * (sgi if sgi is not None else 1.0)
         + 0.115 * (depi if depi is not None else 1.0)
         - 0.172 * (sgai if sgai is not None else 1.0)
         + 4.679 * (tata if tata is not None else 0.0)
         - 0.327 * (lvgi if lvgi is not None else 1.0))

    if np.isnan(m) or np.isinf(m):
        return None
    return round(m, 3)


def beneish_regime(m):
    """Two-bucket Beneish: manipulator vs non-manipulator."""
    if m is None:
        return 'unknown'
    # -1.78 is canonical 1999 threshold
    if m > -1.78:
        return 'manipulator'
    return 'clean'


# ──────────────────────────────────────────────────────────────────────────────
# Composite regime
# ──────────────────────────────────────────────────────────────────────────────

def composite_regime(isc_regime, altman_reg, piotroski_reg, beneish_reg):
    """
    Combine four signals into a single 4-bucket regime.

    Each framework contributes a "stress level" 0-3:
        ISC:        stable=0, elevated=1, rising=2, distressed=3
        Altman:     safe=0,   grey=1,    distress=3
        Piotroski:  strong=0, mixed=1,   weak=3
        Beneish:    clean=0,  manipulator=2
    Then average, and re-bucket to stable/elevated/rising/distressed.
    """
    isc_score = {'stable': 0, 'elevated': 1, 'rising': 2, 'distressed': 3}.get(isc_regime)
    alt_score = {'safe': 0, 'grey': 1, 'distress': 3}.get(altman_reg)
    pio_score = {'strong': 0, 'mixed': 1, 'weak': 3}.get(piotroski_reg)
    ben_score = {'clean': 0, 'manipulator': 2}.get(beneish_reg)

    scores = [s for s in [isc_score, alt_score, pio_score, ben_score] if s is not None]
    if len(scores) < 2:
        return 'unknown'
    avg = sum(scores) / len(scores)

    if avg < 0.75:
        return 'stable'
    if avg < 1.5:
        return 'elevated'
    if avg < 2.25:
        return 'rising'
    return 'distressed'
