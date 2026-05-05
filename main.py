"""
ISC Analyst+ Backend API
FastAPI app — deploy on Render free tier
Ryan W. Malone — Independent Researcher — The Filter Lab LLC
doi.org/10.5281/zenodo.18940081
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YF = True
except:
    HAS_YF = False

# Variance EWS score — paper-validated metric (replaces deprecated coupling C)
from variance_score import compute_variance_score, spearman_rho

app = FastAPI(
    title="ISC Analyst+ API",
    description="Full financial review powered by ISC coupling framework",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEADERS = {'User-Agent': 'ISCAnalyst malonrw@gmail.com'}

# ── Tag map ────────────────────────────────────────────────────────────────────
TAG_MAP = {
    'revenue':            ['Revenues','RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','NetRevenues'],
    'gross_profit':       ['GrossProfit'],
    'operating_income':   ['OperatingIncomeLoss'],
    'net_income':         ['NetIncomeLoss','ProfitLoss'],
    'interest_expense':   ['InterestExpense','InterestAndDebtExpense'],
    'da':                 ['DepreciationDepletionAndAmortization','DepreciationAndAmortization'],
    'income_tax':         ['IncomeTaxExpenseBenefit'],
    'eps_diluted':        ['EarningsPerShareDiluted'],
    'total_assets':       ['Assets'],
    'current_assets':     ['AssetsCurrent'],
    'cash':               ['CashAndCashEquivalentsAtCarryingValue','CashCashEquivalentsAndShortTermInvestments'],
    'receivables':        ['AccountsReceivableNetCurrent'],
    'inventory':          ['InventoryNet','Inventories'],
    'ppe_net':            ['PropertyPlantAndEquipmentNet'],
    'goodwill':           ['Goodwill'],
    'total_liabilities':  ['Liabilities'],
    'current_liabilities':['LiabilitiesCurrent'],
    'accounts_payable':   ['AccountsPayableCurrent'],
    'short_term_debt':    ['ShortTermBorrowings','DebtCurrent'],
    'long_term_debt':     ['LongTermDebt','LongTermDebtNoncurrent'],
    'total_equity':       ['StockholdersEquity','StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'retained_earnings':  ['RetainedEarningsAccumulatedDeficit'],
    'cfo':                ['NetCashProvidedByUsedInOperatingActivities','NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
    'capex':              ['PaymentsToAcquirePropertyPlantAndEquipment'],
    'cfi':                ['NetCashProvidedByUsedInInvestingActivities'],
    'cff':                ['NetCashProvidedByUsedInFinancingActivities'],
    'shares_outstanding': ['CommonStockSharesOutstanding'],
}

# ── EDGAR helpers ──────────────────────────────────────────────────────────────
def get_cik(ticker: str):
    try:
        r = requests.get(
            'https://www.sec.gov/files/company_tickers.json',
            headers=HEADERS, timeout=15
        )
        data = r.json()
        for entry in data.values():
            if entry['ticker'].upper() == ticker.upper():
                return str(entry['cik_str']).zfill(10)
    except Exception as e:
        pass
    return None

def get_facts(cik: str):
    try:
        r = requests.get(
            f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
            headers=HEADERS, timeout=30
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def extract_series(facts, key, n=20):
    if not facts or 'us-gaap' not in facts.get('facts', {}):
        return pd.Series(dtype=float)
    usgaap = facts['facts']['us-gaap']
    for tag in TAG_MAP.get(key, []):
        if tag not in usgaap:
            continue
        units = usgaap[tag].get('units', {})
        for unit in ['USD', 'shares', 'USD/shares']:
            if unit not in units:
                continue
            entries = [e for e in units[unit] if 'end' in e and 'val' in e]
            if not entries:
                continue
            seen = {}
            for e in entries:
                k = e['end']
                if k not in seen or e.get('filed','') > seen[k].get('filed',''):
                    seen[k] = e
            sorted_e = sorted(seen.values(), key=lambda x: x['end'])[-n:]
            if len(sorted_e) < 4:
                continue
            div = 1e6 if unit == 'USD' else 1
            s = pd.Series(
                [e['val']/div for e in sorted_e],
                index=pd.to_datetime([e['end'] for e in sorted_e])
            )
            return s[~s.index.duplicated(keep='last')]
    return pd.Series(dtype=float)

# ── Per-series trajectory (replaces deprecated compute_coupling) ──────────────
# 
# DEPRECATED: compute_coupling() previously computed C = corr(Var_W, AR1_W) on
# quarterly fundamental series. Per Malone 2026 (Filter Collapse paper, P07),
# this metric has near-zero correlation in finance (r=0.039, CI [0.035,0.043],
# N=8,785) and joint variance/AR1 criteria yield AUC=0.51-0.60 in finance vs
# variance-only AUC=0.86-0.88. The coupling metric was both statistically
# meaningless on n=4-20 quarterly samples AND paper-disconfirmed for finance.
#
# The product now uses compute_variance_score() (price-based, paper-validated,
# AUC=0.86-0.96) for the primary structural signal. Per-series fundamentals are
# preserved here for trajectory and level analysis only — no C correlation.

def compute_series_trajectory(series, window=6):
    """
    Analyze a quarterly fundamental series without computing the deprecated
    coupling C correlation. Returns trajectory metrics that ARE statistically
    meaningful on small quarterly samples.
    
    Returns:
      {
        'latest':      most recent observed value,
        'trend_rho':   Spearman correlation with time (-1 to +1),
        'pct_change':  total percent change first to last,
        'avg':         mean across the series,
        'n':           number of observations,
        'direction':   'rising' | 'falling' | 'flat' (based on rho)
      }
    Returns None if insufficient data.
    """
    s = series.dropna() if series is not None else pd.Series()
    if len(s) < 4:
        return None
    
    values = s.values
    times = np.arange(len(values))
    
    # Trend: Spearman rho (well-defined on small samples)
    rho = spearman_rho(times, values)
    
    # Direction
    if np.isnan(rho):
        direction = 'flat'
    elif rho > 0.4:
        direction = 'rising'
    elif rho < -0.4:
        direction = 'falling'
    else:
        direction = 'flat'
    
    first_val = float(values[0]) if values[0] != 0 else None
    last_val = float(values[-1])
    pct_change = round(((last_val / first_val) - 1.0) * 100, 1) if first_val and first_val != 0 else None
    
    return {
        'latest':     round(last_val, 2),
        'trend_rho':  round(float(rho), 3) if not np.isnan(rho) else None,
        'pct_change': pct_change,
        'avg':        round(float(np.mean(values)), 2),
        'n':          len(values),
        'direction':  direction,
    }

# ── Market data ────────────────────────────────────────────────────────────────
def get_market_data(ticker: str):
    if not HAS_YF:
        return {}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        # Fetch 2y for variance score (needs 252+ trading days), tail(60) shown in chart
        hist = tk.history(period='2y', auto_adjust=True)
        price   = info.get('currentPrice') or info.get('regularMarketPrice')
        high52  = info.get('fiftyTwoWeekHigh')
        low52   = info.get('fiftyTwoWeekLow')
        mktcap  = info.get('marketCap')

        rsi = None
        if len(hist) >= 15:
            delta = hist['Close'].diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs    = gain / loss
            val   = 100 - 100/(1+rs.iloc[-1])
            rsi   = round(float(val), 1) if not np.isnan(val) else None

        price_hist = []
        full_price_series = None  # for variance score
        if len(hist) > 0:
            price_hist = [round(float(v), 2) for v in hist['Close'].tail(60).tolist() if not np.isnan(v)]
            # Full series with date index for variance EWS computation
            full_price_series = hist['Close'].copy()

        atm_iv = otm_iv = None
        try:
            expiries = tk.options
            if expiries and price:
                chain = tk.option_chain(expiries[0])
                calls = chain.calls.copy()
                puts  = chain.puts.copy()
                if not calls.empty:
                    calls['dist'] = abs(calls['strike'] - price)
                    iv = calls.loc[calls['dist'].idxmin(), 'impliedVolatility']
                    atm_iv = round(float(iv)*100, 1) if not np.isnan(float(iv)) else None
                if not puts.empty and price:
                    puts['dist'] = abs(puts['strike'] - price*0.90)
                    iv2 = puts.loc[puts['dist'].idxmin(), 'impliedVolatility']
                    otm_iv = round(float(iv2)*100, 1) if not np.isnan(float(iv2)) else None
        except:
            pass

        def safe_round(v, d=2):
            try:
                f = float(v)
                return round(f, d) if not np.isnan(f) and not np.isinf(f) else None
            except:
                return None

        pct52 = None
        if all([price, high52, low52]) and high52 != low52:
            pct52 = round(((price-low52)/(high52-low52))*100, 1)

        return {
            'price':         safe_round(price, 2),
            'high_52w':      safe_round(high52, 2),
            'low_52w':       safe_round(low52, 2),
            'pct_52w':       pct52,
            'market_cap':    mktcap,
            'market_cap_bn': safe_round(mktcap/1e9, 2) if mktcap else None,
            'market_cap_m':  safe_round(mktcap/1e6, 1) if mktcap else None,
            'pe':            safe_round(info.get('trailingPE'), 1),
            'pb':            safe_round(info.get('priceToBook'), 2),
            'ev_ebitda':     safe_round(info.get('enterpriseToEbitda'), 1),
            'beta':          safe_round(info.get('beta'), 2),
            'rsi':           rsi,
            'sector':        info.get('sector'),
            'industry':      info.get('industry'),
            'company_name':  info.get('longName') or info.get('shortName') or ticker,
            'description':   (info.get('longBusinessSummary') or '')[:300],
            'price_history': price_hist,
            'full_price_series': full_price_series,  # for variance score; not in JSON output
            'atm_iv':        atm_iv,
            'otm_iv':        otm_iv,
            'iv_skew':       round(otm_iv-atm_iv, 1) if (atm_iv and otm_iv) else None,
            'total_revenue': safe_round(info.get('totalRevenue'), 0),
            'ebitda':        safe_round(info.get('ebitda'), 0),
            'total_debt':    safe_round(info.get('totalDebt'), 0),
            'free_cashflow': safe_round(info.get('freeCashflow'), 0),
            'operating_cashflow': safe_round(info.get('operatingCashflow'), 0),
            'current_ratio': safe_round(info.get('currentRatio'), 2),
            'quick_ratio':   safe_round(info.get('quickRatio'), 2),
            'debt_to_equity':safe_round(info.get('debtToEquity'), 2),
            'roe':           safe_round(info.get('returnOnEquity'), 4),
            'roa':           safe_round(info.get('returnOnAssets'), 4),
            'gross_margins': safe_round(info.get('grossMargins'), 4),
            'op_margins':    safe_round(info.get('operatingMargins'), 4),
            'profit_margins':safe_round(info.get('profitMargins'), 4),
            'revenue_growth':safe_round(info.get('revenueGrowth'), 4),
            'earnings_growth':safe_round(info.get('earningsGrowth'), 4),
            'shares_outstanding': info.get('sharesOutstanding'),
            'dividend_yield': safe_round(info.get('dividendYield'), 4),
            'eps_trailing':  safe_round(info.get('trailingEps'), 2),
            'book_value':    safe_round(info.get('bookValue'), 2),
        }
    except Exception as e:
        return {'error': str(e)}

# ── Traditional metrics ────────────────────────────────────────────────────────
def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    v = a / b
    return round(v, 3) if not (np.isnan(v) or np.isinf(v)) else None

def last(s):
    if s is None or len(s) == 0:
        return None
    v = s.dropna()
    if len(v) == 0:
        return None
    val = float(v.iloc[-1])
    return round(val, 3) if not (np.isnan(val) or np.isinf(val)) else None

def ttm(s):
    if s is None or len(s) == 0:
        return None
    v = s.dropna()
    if len(v) < 4:
        return None
    val = float(v.iloc[-4:].sum())
    return round(val, 3) if not (np.isnan(val) or np.isinf(val)) else None

def is_trend_up(s, n=4):
    if s is None or len(s.dropna()) < n+1:
        return None
    v = s.dropna()
    return float(v.iloc[-1]) > float(v.iloc[-n-1])

def compute_altman_z(ta, re, ebit, rev, tl, ca, cl, mktcap_m):
    try:
        if not ta or ta == 0 or not tl or tl == 0:
            return None
        wc = (ca or 0) - (cl or 0)
        X1 = wc / ta
        X2 = (re or 0) / ta
        X3 = (ebit or 0) / ta
        X4 = (mktcap_m or ta*1.2) / tl
        X5 = (rev or 0) / ta
        z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        return round(z, 3) if not (np.isnan(z) or np.isinf(z)) else None
    except:
        return None

def compute_piotroski(roa, cfo, ni, roa_up, leverage_down, liquidity_up, shares_up, margin_up, turnover_up):
    signals = {
        'ROA positive':       1 if (roa or 0) > 0 else 0,
        'CFO positive':       1 if (cfo or 0) > 0 else 0,
        'ROA improving':      1 if roa_up else 0,
        'CFO > Net Income':   1 if (cfo or 0) > (ni or 0) else 0,
        'Leverage falling':   1 if leverage_down else 0,
        'Liquidity rising':   1 if liquidity_up else 0,
        'No dilution':        1 if not shares_up else 0,
        'Margin improving':   1 if margin_up else 0,
        'Turnover improving': 1 if turnover_up else 0,
    }
    return sum(signals.values()), signals

def rate_metric(val, metric):
    """Return signal color and label for a metric value."""
    rules = {
        'altman_z':        [(1.81,'red','✗ Distress zone'),(3.0,'amber','⚠ Grey zone'),(99,'green','✓ Safe zone')],
        'piotroski_f':     [(3,'red','✗ Weak'),(7,'amber','~ Neutral'),(9,'green','✓ Strong')],
        'current_ratio':   [(1.0,'red','✗ Below 1 — stress'),(1.5,'amber','⚠ Tight'),(9,'green','✓ Healthy')],
        'interest_cov':    [(1.5,'red','✗ Stress'),(3.0,'amber','⚠ Watch'),(99,'green','✓ Healthy')],
        'debt_ebitda':     [(2.0,'green','✓ Conservative'),(4.0,'amber','~ Moderate'),(99,'red','✗ Elevated')],
        # variance_regime: rates the variance EWS score (annualized rolling variance)
        # Calibrated against paper distributions: stable ~0.04-0.10, distress 0.15-0.50+
        'variance_regime': [(0.10,'green','✓ Stable'),(0.25,'amber','⚠ Elevated'),(99,'red','✗ Distressed')],
    }
    r = rules.get(metric, [])
    if val is None:
        return 'slate', '—'
    for threshold, color, label in r:
        if val < threshold:
            return color, label
    return 'slate', '—'

# ── Clean JSON ─────────────────────────────────────────────────────────────────
def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj

# ── Main analysis endpoint ─────────────────────────────────────────────────────
@app.get("/analyze/{ticker}")
async def analyze(ticker: str, window: int = 6):
    ticker = ticker.upper().strip()

    # 1. Market data (yfinance)
    mkt = get_market_data(ticker)
    if 'error' in mkt and not mkt.get('price'):
        # yfinance failed — try EDGAR only
        pass

    # 2. EDGAR
    cik = get_cik(ticker)
    facts = get_facts(cik) if cik else None

    # 3. Extract series from EDGAR
    raw = {}
    if facts:
        for key in TAG_MAP:
            raw[key] = extract_series(facts, key)

    # 4. Derive TTM values
    rev_ttm   = ttm(raw.get('revenue'))   or (mkt.get('total_revenue', 0) / 1e6 if mkt.get('total_revenue') else None)
    gp_ttm    = ttm(raw.get('gross_profit'))
    oi_ttm    = ttm(raw.get('operating_income'))
    ni_ttm    = ttm(raw.get('net_income'))
    int_ttm   = ttm(raw.get('interest_expense'))
    da_ttm    = ttm(raw.get('da'))
    cfo_ttm   = ttm(raw.get('cfo'))       or (mkt.get('operating_cashflow', 0) / 1e6 if mkt.get('operating_cashflow') else None)
    capex_ttm = ttm(raw.get('capex'))
    cfi_ttm   = ttm(raw.get('cfi'))
    cff_ttm   = ttm(raw.get('cff'))

    ebitda_ttm = ttm(raw.get('operating_income'))
    if ebitda_ttm and da_ttm:
        ebitda_ttm = round(ebitda_ttm + da_ttm, 2)
    if not ebitda_ttm:
        ebitda_ttm = mkt.get('ebitda', 0) / 1e6 if mkt.get('ebitda') else None

    fcf_ttm = None
    if cfo_ttm is not None and capex_ttm is not None:
        fcf_ttm = round(cfo_ttm - abs(capex_ttm), 2)
    elif mkt.get('free_cashflow'):
        fcf_ttm = round(mkt['free_cashflow'] / 1e6, 2)

    ta  = last(raw.get('total_assets'))
    ca  = last(raw.get('current_assets'))
    cl  = last(raw.get('current_liabilities'))
    cash = last(raw.get('cash'))
    rec  = last(raw.get('receivables'))
    inv  = last(raw.get('inventory'))
    ltd  = last(raw.get('long_term_debt'))
    std  = last(raw.get('short_term_debt'))
    tl   = last(raw.get('total_liabilities'))
    te   = last(raw.get('total_equity'))
    re   = last(raw.get('retained_earnings'))
    ap   = last(raw.get('accounts_payable'))
    ppe  = last(raw.get('ppe_net'))
    gw   = last(raw.get('goodwill'))

    total_debt = None
    if ltd is not None or std is not None:
        total_debt = round((ltd or 0) + (std or 0), 2)
    if not total_debt and mkt.get('total_debt'):
        total_debt = round(mkt['total_debt'] / 1e6, 2)

    wc = round(ca - cl, 2) if (ca is not None and cl is not None) else None

    # Margins
    gross_margin  = safe_div(gp_ttm, rev_ttm)
    if gross_margin is None and mkt.get('gross_margins'):
        gross_margin = mkt['gross_margins']
    op_margin     = safe_div(oi_ttm, rev_ttm) or mkt.get('op_margins')
    net_margin    = safe_div(ni_ttm, rev_ttm) or mkt.get('profit_margins')
    fcf_margin    = safe_div(fcf_ttm, rev_ttm)
    roa           = safe_div(ni_ttm, ta)      or mkt.get('roa')
    roe           = safe_div(ni_ttm, te)      or mkt.get('roe')
    curr_ratio    = safe_div(ca, cl)          or mkt.get('current_ratio')
    quick_ratio   = safe_div((ca or 0)-(inv or 0), cl) if (ca and cl) else mkt.get('quick_ratio')
    de_ratio      = safe_div(total_debt, te)  or mkt.get('debt_to_equity')
    debt_ebitda   = safe_div(total_debt, ebitda_ttm)
    int_coverage  = safe_div(oi_ttm, int_ttm)
    asset_turn    = safe_div(rev_ttm, ta)
    cash_conv     = safe_div(cfo_ttm, ni_ttm)

    # Altman Z
    altman = compute_altman_z(ta, re, oi_ttm, rev_ttm, tl, ca, cl, mkt.get('market_cap_m'))

    # Piotroski
    f_score, f_signals = compute_piotroski(
        roa, cfo_ttm, ni_ttm,
        is_trend_up(raw.get('net_income')),
        is_trend_up(raw.get('long_term_debt')) == False,
        is_trend_up(raw.get('current_assets')),
        is_trend_up(raw.get('shares_outstanding')),
        is_trend_up(raw.get('gross_profit')),
        is_trend_up(raw.get('revenue')),
    )

    # 5. Structural EWS — variance-based score (paper-validated AUC = 0.86-0.96)
    #    Computed from daily price returns, NOT quarterly fundamentals.
    #    Reference: Malone 2026 (Filter Collapse paper P07; IRFA submission).
    variance_score = None
    if mkt.get('full_price_series') is not None:
        variance_score = compute_variance_score(
            mkt['full_price_series'],
            window_days=252,
            rolling_window=90,
        )

    # Per-series fundamental trajectory (informational, not used for primary scoring)
    isc_series = {
        'Revenue':              raw.get('revenue'),
        'Gross Profit':         raw.get('gross_profit'),
        'Operating Income':     raw.get('operating_income'),
        'Net Income':           raw.get('net_income'),
        'Operating Cash Flow':  raw.get('cfo'),
        'Total Assets':         raw.get('total_assets'),
        'Long-Term Debt':       raw.get('long_term_debt'),
        'Cash':                 raw.get('cash'),
        'Interest Expense':     raw.get('interest_expense'),
    }
    series_trajectories = {}
    for name, s in isc_series.items():
        traj = compute_series_trajectory(s)
        if traj:
            series_trajectories[name] = traj

    # Primary structural signal — variance score if available, otherwise unavailable
    if variance_score and 'error' not in variance_score:
        primary_variance = variance_score.get('mean_variance')
        primary_trend    = variance_score.get('variance_trend')
        primary_ratio    = variance_score.get('variance_ratio')
        primary_regime   = variance_score.get('regime')   # stable | elevated | rising | distressed
        score_available  = True
        score_error      = None
    else:
        primary_variance = None
        primary_trend    = None
        primary_ratio    = None
        primary_regime   = 'unavailable'
        score_available  = False
        score_error      = (variance_score.get('error') if variance_score else 'no_price_data')

    # 6. Options — IV multiplier mapped to new regime labels
    atm_iv    = mkt.get('atm_iv')
    mult_map  = {
        'stable':      0.92,
        'elevated':    1.05,
        'rising':      1.18,
        'distressed':  1.35,
        'unavailable': 1.0,
    }
    mult      = mult_map.get(primary_regime, 1.0)
    fair_iv   = round(atm_iv * mult, 1) if atm_iv else None
    iv_gap    = round(fair_iv - atm_iv, 1) if (fair_iv and atm_iv) else None

    # 7. Ratings
    def R(val, key):
        color, label = rate_metric(val, key)
        return {'color': color, 'label': label}

    # 8. Plain-English summaries
    def plain_variance(score, trend, ratio, regime, ticker):
        if regime == 'unavailable':
            return f'Variance EWS unavailable — yfinance price data could not be retrieved for {ticker}. Traditional metrics (Altman Z, Piotroski F) below remain valid.'
        if score is None:
            return f'Insufficient price history to compute variance EWS for {ticker}.'
        # Variance score in annualized units (e.g., 0.10 = ~32% annualized vol)
        vol_pct = round((score ** 0.5) * 100, 1)  # annualized stddev as %
        trend_label = ('rising' if (trend or 0) > 0.4 else
                       'falling' if (trend or 0) < -0.4 else 'flat')
        texts = {
            'stable':     f'{ticker} shows stable equity dynamics. Mean rolling variance = {score:.4f} (annualized vol ~{vol_pct}%); trend {trend_label}. Consistent with paper baseline for non-distressed firms (variance ~0.04-0.10). No structural red flags from the variance EWS.',
            'elevated':   f'{ticker} shows elevated equity volatility. Mean rolling variance = {score:.4f} (annualized vol ~{vol_pct}%); trend {trend_label}. Above the typical stable-firm baseline. Worth comparing to sector peers and watching the trend.',
            'rising':     f'{ticker} shows rising volatility — the trend signal that historically precedes structural transitions. Mean variance = {score:.4f} (annualized vol ~{vol_pct}%); trend rho = {trend:.2f}; latest/baseline ratio = {ratio:.2f}x. The paper-validated EWS is active; this pattern preceded events in Lehman, AIG, SVB, and others.',
            'distressed': f'{ticker} shows high volatility AND a rising trend — the strongest variance EWS signal. Mean variance = {score:.4f} (annualized vol ~{vol_pct}%); trend rho = {trend:.2f}; latest/baseline ratio = {ratio:.2f}x. This combination of magnitude and trajectory matches the paper\'s pre-collapse reference cases.',
        }
        return texts.get(regime, f'Variance EWS regime: {regime}.')

    def plain_z(z):
        if z is None: return 'Could not compute — missing balance sheet data.'
        if z < 1.81: return f'Altman Z = {z:.2f}. In the distress zone (below 1.81). Historically, companies here have had high rates of financial distress within 2 years. Not a guarantee — but a significant flag.'
        if z < 3.0:  return f'Altman Z = {z:.2f}. In the grey zone (1.81–3.0). Mixed signals. Could go either way. Worth watching alongside other metrics.'
        return f'Altman Z = {z:.2f}. In the safe zone (above 3.0). Financially healthy by this traditional measure.'

    def plain_piotroski(f):
        if f is None: return 'Could not compute.'
        if f <= 2: return f'Piotroski F = {f}/9. Failing most of the 9 financial health tests. Weak across profitability, leverage, and efficiency.'
        if f <= 6: return f'Piotroski F = {f}/9. Mixed results. Passing some tests, failing others.'
        return f'Piotroski F = {f}/9. Passing most financial health tests. Strong signal across profitability, leverage, and efficiency.'

    def divergence_summary(score, regime, altman, ticker):
        if regime == 'unavailable' or altman is None:
            return 'Insufficient data for divergence analysis.'
        # Variance EWS healthy = stable; not healthy = elevated/rising/distressed
        ews_ok = regime == 'stable'
        alt_ok = altman >= 3.0
        if ews_ok and alt_ok:
            return f'Both the variance EWS (regime: {regime}) and Altman Z ({altman:.2f}) are consistent — structural health confirmed for {ticker}. No divergence.'
        if not ews_ok and not alt_ok:
            return f'Both signals are flagging stress (variance regime: {regime}; Altman Z = {altman:.2f}). Signals converging — multiple metrics now consistent.'
        if not ews_ok and alt_ok:
            return f'Variance EWS LEADING — early window potentially active. The variance regime is {regime} (mean variance = {score:.4f}) while Altman Z ({altman:.2f}) still looks OK. Historical lead time on this divergence pattern: 1-5 quarters in retrospective testing. This is the core EWS value proposition.'
        return f'Variance EWS shows stable but Altman Z ({altman:.2f}) is flagging. Possible accounting/balance-sheet stress not yet reflected in equity dynamics. Worth investigating further.'

    # ── Build response ─────────────────────────────────────────────────────────
    result = {
        'ticker':       ticker,
        'company_name': mkt.get('company_name', ticker),
        'sector':       mkt.get('sector'),
        'industry':     mkt.get('industry'),
        'description':  mkt.get('description', ''),
        'analysis_date':str(pd.Timestamp.now().date()),
        'data_sources': {
            'edgar': facts is not None,
            'yfinance': bool(mkt.get('price')),
        },

        'isc': {
            # === New variance-based primary signal (paper-validated) ===
            'variance_score':    primary_variance,        # mean rolling 90d variance, annualized
            'variance_trend':    primary_trend,           # Spearman rho of variance with time
            'variance_ratio':    primary_ratio,           # latest / baseline
            'regime':            primary_regime,          # stable | elevated | rising | distressed | unavailable
            'available':         score_available,
            'error':             score_error,
            # === Per-series fundamental trajectory (informational only, no C) ===
            'by_series':         series_trajectories,
            # === Display ===
            'rating':            R(primary_variance, 'variance_regime') if primary_variance is not None else {'color':'slate','label':'—'},
            'plain':             plain_variance(primary_variance, primary_trend, primary_ratio, primary_regime, ticker),
            'explain':           'Variance EWS score: mean rolling 90-day variance of daily log returns over a 252-day window, annualized. Regimes calibrated against Malone 2026 (Filter Collapse, Zenodo 18940081). Stable: <0.10. Elevated: 0.10-0.25. Distressed: >0.25. Trend (Spearman rho with time) and ratio (latest/baseline) determine rising vs static elevation. AUC = 0.86-0.96 on validation set of 45 collapse + 200 stable windows (paper IRFA submission).',
            'methodology':       'price_based_variance_ews_v1',
            # === Backward-compat aliases (will be removed in next major version) ===
            'C':                 primary_variance,        # alias: legacy frontends may read 'C'
            'trajectory':        [],                      # deprecated; was the rolling C trajectory
        },

        'income_statement': {
            'revenue':          {'val': rev_ttm,    'label': 'Revenue (TTM $M)',          'simple': 'Total sales — money coming in the front door'},
            'gross_profit':     {'val': gp_ttm,     'label': 'Gross Profit (TTM $M)',      'simple': 'Revenue minus direct cost of making the product'},
            'gross_margin':     {'val': round((gross_margin or 0)*100, 1) if gross_margin else None, 'label': 'Gross Margin %', 'simple': 'What fraction of each sale is kept after direct costs'},
            'operating_income': {'val': oi_ttm,     'label': 'Operating Income (TTM $M)',  'simple': 'Profit from running the business — before debt and taxes'},
            'operating_margin': {'val': round((op_margin or 0)*100, 1) if op_margin else None, 'label': 'Operating Margin %', 'simple': 'Operating profit as % of sales'},
            'ebitda':           {'val': ebitda_ttm, 'label': 'EBITDA (TTM $M)',             'simple': 'Cash-like profit before accounting adjustments'},
            'net_income':       {'val': ni_ttm,     'label': 'Net Income (TTM $M)',         'simple': 'What the company kept after every cost — the bottom line'},
            'net_margin':       {'val': round((net_margin or 0)*100, 1) if net_margin else None, 'label': 'Net Margin %', 'simple': 'Final profit as % of sales'},
            'interest_expense': {'val': int_ttm,    'label': 'Interest Expense (TTM $M)',   'simple': 'What the company pays on its debt each year'},
            'da':               {'val': da_ttm,     'label': 'D&A (TTM $M)',               'simple': 'Paper cost of wearing out assets — not actual cash going out'},
            'eps':              {'val': mkt.get('eps_trailing'), 'label': 'EPS (Trailing)', 'simple': 'Profit per share — what each share earned'},
            'revenue_growth':   {'val': round((mkt.get('revenue_growth') or 0)*100, 1) if mkt.get('revenue_growth') else None, 'label': 'Revenue Growth YoY %', 'simple': 'How fast is the top line growing?'},
        },

        'balance_sheet': {
            'total_assets':        {'val': ta,   'label': 'Total Assets ($M)',          'simple': 'Everything the company owns'},
            'cash':                {'val': cash, 'label': 'Cash ($M)',                  'simple': 'Money in the bank right now'},
            'current_assets':      {'val': ca,   'label': 'Current Assets ($M)',        'simple': 'Will turn into cash within 12 months'},
            'receivables':         {'val': rec,  'label': 'Accounts Receivable ($M)',   'simple': 'Money customers owe but have not paid'},
            'inventory':           {'val': inv,  'label': 'Inventory ($M)',             'simple': 'Products made but not sold yet'},
            'ppe_net':             {'val': ppe,  'label': 'PP&E Net ($M)',              'simple': 'Buildings, machines, equipment after depreciation'},
            'goodwill':            {'val': gw,   'label': 'Goodwill ($M)',              'simple': 'Premium paid in past acquisitions'},
            'total_liabilities':   {'val': tl,   'label': 'Total Liabilities ($M)',     'simple': 'Everything the company owes'},
            'current_liabilities': {'val': cl,   'label': 'Current Liabilities ($M)',   'simple': 'Bills due in the next 12 months'},
            'long_term_debt':      {'val': ltd,  'label': 'Long-Term Debt ($M)',        'simple': 'Loans and bonds due more than a year from now'},
            'total_debt':          {'val': total_debt,'label': 'Total Debt ($M)',       'simple': 'All short and long-term borrowing combined'},
            'accounts_payable':    {'val': ap,   'label': 'Accounts Payable ($M)',      'simple': 'Bills owed to suppliers not yet paid'},
            'total_equity':        {'val': te,   'label': "Shareholders' Equity ($M)",  'simple': 'What would be left for shareholders if everything was liquidated'},
            'retained_earnings':   {'val': re,   'label': 'Retained Earnings ($M)',     'simple': 'Accumulated profits kept in the business over all years'},
            'working_capital':     {'val': wc,   'label': 'Working Capital ($M)',       'simple': 'Current assets minus current liabilities — the short-term cushion'},
        },

        'cash_flow': {
            'cfo':           {'val': cfo_ttm,   'label': 'Operating Cash Flow (TTM $M)', 'simple': 'Actual cash the business generates — harder to fake than net income'},
            'capex':         {'val': capex_ttm, 'label': 'CapEx (TTM $M)',               'simple': 'Cash spent buying or maintaining physical assets'},
            'fcf':           {'val': fcf_ttm,   'label': 'Free Cash Flow (TTM $M)',       'simple': 'Cash left after running the business — the gold standard'},
            'cfi':           {'val': cfi_ttm,   'label': 'Investing Cash Flow (TTM $M)', 'simple': 'Cash spent or received from buying and selling assets'},
            'cff':           {'val': cff_ttm,   'label': 'Financing Cash Flow (TTM $M)', 'simple': 'Cash from borrowing, repaying debt, issuing or buying back stock'},
            'fcf_margin':    {'val': round((fcf_margin or 0)*100, 1) if fcf_margin else None, 'label': 'FCF Margin %', 'simple': 'Free cash flow as % of revenue'},
            'cash_conversion':{'val': cash_conv,'label': 'Cash Conversion (CFO/NI)',     'simple': 'Is reported profit backed by actual cash? Above 1.0 is good'},
        },

        'traditional': {
            'altman_z':        {'val': altman,      'label': 'Altman Z-Score',     'simple': 'Classic 5-ratio bankruptcy predictor. Above 3 safe. Below 1.81 distress.',    'rating': R(altman, 'altman_z'),      'plain': plain_z(altman)},
            'piotroski_f':     {'val': f_score,     'label': 'Piotroski F-Score',  'simple': '9-question health checklist. 0-9 scale. Above 7 strong. Below 3 weak.',        'rating': R(f_score, 'piotroski_f'),  'plain': plain_piotroski(f_score), 'signals': f_signals},
            'current_ratio':   {'val': curr_ratio,  'label': 'Current Ratio',      'simple': 'Can we pay our 12-month bills? Above 1.5 is healthy. Below 1.0 is stress.',    'rating': R(curr_ratio, 'current_ratio')},
            'quick_ratio':     {'val': quick_ratio, 'label': 'Quick Ratio',        'simple': 'Can we pay bills without selling inventory? Tougher liquidity test.'},
            'interest_coverage':{'val': int_coverage,'label': 'Interest Coverage', 'simple': 'How many times over can earnings cover interest payments? Above 3 is healthy.', 'rating': R(int_coverage, 'interest_cov')},
            'debt_to_ebitda':  {'val': debt_ebitda, 'label': 'Debt / EBITDA',      'simple': 'How many years of earnings to pay off all debt? Under 3 is healthy.',          'rating': R(debt_ebitda, 'debt_ebitda')},
            'debt_to_equity':  {'val': de_ratio,    'label': 'Debt / Equity',      'simple': 'How much debt relative to shareholder value? Higher means more leveraged.'},
            'roa':             {'val': round((roa or 0)*100,2) if roa else None, 'label': 'ROA %', 'simple': 'How efficiently does the company use its assets to make money?'},
            'roe':             {'val': round((roe or 0)*100,2) if roe else None, 'label': 'ROE %', 'simple': 'How much profit per dollar shareholders have invested?'},
            'asset_turnover':  {'val': asset_turn,  'label': 'Asset Turnover',     'simple': 'How much revenue does each dollar of assets generate?'},
            'gross_margin_pct':{'val': round((gross_margin or 0)*100,1) if gross_margin else None, 'label': 'Gross Margin %', 'simple': 'Fraction of each sale kept after direct costs'},
            'op_margin_pct':   {'val': round((op_margin or 0)*100,1) if op_margin else None,   'label': 'Operating Margin %', 'simple': 'Operating profit as fraction of sales'},
        },

        'market': {
            'price':         mkt.get('price'),
            'high_52w':      mkt.get('high_52w'),
            'low_52w':       mkt.get('low_52w'),
            'pct_52w':       mkt.get('pct_52w'),
            'market_cap_bn': mkt.get('market_cap_bn'),
            'pe':            mkt.get('pe'),
            'pb':            mkt.get('pb'),
            'ev_ebitda':     mkt.get('ev_ebitda'),
            'beta':          mkt.get('beta'),
            'rsi':           mkt.get('rsi'),
            'dividend_yield':round((mkt.get('dividend_yield') or 0)*100, 2) if mkt.get('dividend_yield') else None,
            'eps':           mkt.get('eps_trailing'),
            'book_value':    mkt.get('book_value'),
            'price_history': mkt.get('price_history', []),
            'price_simple':  'Current stock price',
            'pe_simple':     'Price-to-Earnings: what investors pay per $1 of profit. Lower is cheaper.',
            'beta_simple':   'How much the stock moves vs the market. Beta 1.5 = moves 50% more than market.',
            'rsi_simple':    'Momentum indicator 0-100. Above 70 = overbought (may pull back). Below 30 = oversold (may bounce).',
        },

        'options': {
            'atm_iv':      mkt.get('atm_iv'),
            'otm_iv':      mkt.get('otm_iv'),
            'iv_skew':     mkt.get('iv_skew'),
            'fair_iv':     fair_iv,
            'iv_gap':      iv_gap,
            'multiplier':  mult,
            'regime':      primary_regime,
            'rating':      R(iv_gap, 'iv_gap') if iv_gap else {'color':'slate','label':'—'},
            'atm_iv_simple':  'What the options market expects the stock to move over the next year (annualized %)',
            'iv_skew_simple': 'Extra cost of downside protection — high skew means market fears a crash',
            'fair_iv_simple': 'EWS-adjusted volatility estimate based on structural regime',
            'iv_gap_simple':  'Positive gap = options may be underpricing structural risk',
        },

        'divergence': {
            'summary':    divergence_summary(primary_variance, primary_regime, altman, ticker),
            'isc_regime': primary_regime,
            'altman_z':   altman,
            'lead_time':  ('3-5 quarters' if primary_regime == 'distressed' else
                           '2-4 quarters' if primary_regime == 'rising' else
                           '1-3 quarters' if primary_regime == 'elevated' else 'None'),
            'simple':     'The variance EWS detects equity-market evidence of structural stress; Altman Z reflects accounting balance-sheet health. When the EWS leads Altman Z, the gap is the early-warning window — empirically 1-5 quarters in retrospective testing on 45 collapse cases.',
        },
    }

    return JSONResponse(content=clean_json(result))

@app.get("/health")
async def health():
    return {"status": "ok", "yfinance": HAS_YF}

@app.get("/")
async def root():
    return {
        "name": "ISC Analyst+ API",
        "version": "1.0.0",
        "researcher": "Ryan W. Malone, Independent Researcher",
        "doi": "doi.org/10.5281/zenodo.18940081",
        "endpoint": "/analyze/{ticker}",
        "example": "/analyze/AAPL",
    }
