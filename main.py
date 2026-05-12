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

# Daily price data fetcher with Tiingo + Stooq fallback
# (replaces yfinance for variance score input; yfinance kept as 3rd fallback)
from price_data import fetch_daily_prices, fetch_basic_market_metadata

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
    # ── Industrial / generic (works for tech, retail, manufacturing, etc.) ──
    'revenue':            ['Revenues','RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','NetRevenues'],
    'gross_profit':       ['GrossProfit', 'GrossProfitLoss'],
    'cost_of_revenue':    ['CostOfRevenue', 'CostOfGoodsAndServicesSold',
                           'CostOfGoodsSold', 'CostOfServices'],
    'operating_income':   ['OperatingIncomeLoss'],
    'net_income':         ['NetIncomeLoss','ProfitLoss'],
    'interest_expense':   ['InterestExpense','InterestAndDebtExpense'],
    'da':                 ['DepreciationDepletionAndAmortization','DepreciationAndAmortization'],
    'income_tax':         ['IncomeTaxExpenseBenefit'],
    'eps_diluted':        ['EarningsPerShareDiluted'],
    'total_assets':       ['Assets'],
    'current_assets':     ['AssetsCurrent'],
    'cash':               ['CashAndCashEquivalentsAtCarryingValue','CashCashEquivalentsAndShortTermInvestments'],
    'receivables':        ['AccountsReceivableNetCurrent',
                           'ReceivablesNetCurrent',
                           'AccountsReceivableNet',
                           'AccountsAndNotesReceivableNet',
                           'NontradeReceivablesCurrent',
                           'AccountsReceivableTradeNetCurrent'],
    'accounts_payable':   ['AccountsPayableCurrent',
                           'AccountsPayableAndAccruedLiabilitiesCurrent',
                           'AccountsPayable'],
    'inventory':          ['InventoryNet','Inventories'],
    'ppe_net':             ['PropertyPlantAndEquipmentNet'],
    'goodwill':           ['Goodwill'],
    'total_liabilities':  ['Liabilities'],
    'current_liabilities':['LiabilitiesCurrent'],
    'accounts_payable':   ['AccountsPayableCurrent'],
    'short_term_debt':    ['ShortTermBorrowings','DebtCurrent'],
    'long_term_debt':     ['LongTermDebt','LongTermDebtNoncurrent'],
    'total_equity':       ['StockholdersEquity','StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'retained_earnings':  ['RetainedEarningsAccumulatedDeficit'],
    'cfo':                ['NetCashProvidedByUsedInOperatingActivities','NetCashProvidedByUsedInOperatingActivitiesContinuingOperations'],
    'capex':              ['PaymentsToAcquirePropertyPlantAndEquipment',
                           'PaymentsToAcquireProductiveAssets',
                           'PaymentsToAcquireOtherProductiveAssets',
                           'PaymentsForCapitalImprovements'],
    'cfi':                ['NetCashProvidedByUsedInInvestingActivities'],
    'cff':                ['NetCashProvidedByUsedInFinancingActivities'],
    'shares_outstanding': ['CommonStockSharesOutstanding'],

    # ── Bank-specific (Batch 7h.9) ──
    'interest_income':            ['InterestAndDividendIncomeOperating','InterestIncomeOperating','InterestAndFeeIncomeLoansAndLeases'],
    'interest_expense_bank':      ['InterestExpense','InterestExpenseBorrowings'],
    'net_interest_income':        ['InterestIncomeExpenseNet','NetInterestIncome'],
    'noninterest_income':         ['NoninterestIncome'],
    'noninterest_expense':        ['NoninterestExpense'],
    'provision_loan_losses':      ['ProvisionForLoanLeaseAndOtherLosses','ProvisionForLoanAndLeaseLosses','ProvisionForCreditLosses'],
    'loans_receivable':           ['LoansAndLeasesReceivableNetReportedAmount','LoansAndLeasesReceivableNetOfDeferredIncome','FinancingReceivableNetCurrent'],
    'deposits':                   ['Deposits','DepositsTotal','InterestBearingDepositsInBanks'],
    'tier1_capital':              ['Tier1RiskBasedCapital','Tier1Capital','CommonEquityTier1Capital'],
    'risk_weighted_assets':       ['RiskWeightedAssets','RiskBasedCapitalRequiredToBeWellCapitalized'],
    'allowance_loan_losses':      ['LoansAndLeasesReceivableAllowance','FinancingReceivableAllowanceForCreditLosses','AllowanceForLoanAndLeaseLossesWriteOffs'],

    # ── Insurance-specific (Batch 7h.9) ──
    'premiums_earned':            ['PremiumsEarnedNet','PremiumsEarnedNetPropertyAndCasualty','InsurancePremiumsAndOtherConsiderations'],
    'losses_incurred':            ['LiabilityForFuturePolicyBenefitsPeriodIncreaseDecrease','PolicyholderBenefitsAndClaimsIncurredNet','BenefitsLossesAndExpenses'],
    'underwriting_expenses':      ['DeferredPolicyAcquisitionCostAmortizationExpense','OtherUnderwritingExpense'],
    'investment_income':          ['InvestmentIncomeOperating','NetInvestmentIncome','InterestAndDividendIncomeSecuritiesOperating'],
    'investments_held':           ['Investments','AvailableForSaleSecurities','InvestmentsAndCash'],
    'reserves':                   ['LiabilityForFuturePolicyBenefitsAndUnpaidClaimsAndClaimsAdjustmentExpense','LiabilityForUnpaidClaimsAndClaimsAdjustmentExpense'],

    # ── REIT-specific (Batch 7h.9) ──
    'rental_income':              ['OperatingLeasesIncomeStatementLeaseRevenue','RealEstateRevenueNet','RentsAndOtherTenantReimbursements'],
    'real_estate_revenue':        ['RealEstateRevenueNet','Revenues'],
    'noi':                        ['OperatingIncomeLoss'],   # NOI ≈ rental_income − operating_expenses_real_estate
    'real_estate_at_cost':        ['RealEstateInvestmentPropertyAtCost'],
    'real_estate_accumulated_dep':['RealEstateAccumulatedDepreciation'],
    'real_estate_net':            ['RealEstateInvestmentPropertyNet'],
    'mortgages_payable':          ['MortgageLoansOnRealEstate','SecuredDebt','MortgagesPayableNet'],

    # ── Beneish M-Score additions ──
    # SGA: full SG&A line for SGAI ratio. Some firms report combined, some split.
    'sga':                        ['SellingGeneralAndAdministrativeExpense',
                                   'GeneralAndAdministrativeExpense',
                                   'SellingAndMarketingExpense'],
    # Gross PPE: needed for AQI (other non-current assets calc) and DEPI rate
    'gross_ppe':                  ['PropertyPlantAndEquipmentGross'],
    # Accumulated depreciation: used for DEPI = (Dep_t-1 / (Dep_t-1 + GrossPPE_t-1)) / (Dep_t / (Dep_t + GrossPPE_t))
    'accumulated_depreciation':   ['AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'],
}


# ── Sector detection ───────────────────────────────────────────────────────────
# Maps SIC code prefixes (or exact codes) to one of: industrial / bank / insurance / reit / other.
# Source: SEC SIC code list (https://www.sec.gov/info/edgar/siccodes.htm)
def detect_sector(sic_code):
    """Return one of: 'bank', 'insurance', 'reit', 'industrial', 'other'.
    Industrial covers most companies (tech, retail, manufacturing, energy, etc.).
    'other' is for explicitly financial-but-not-bank categories
    (asset managers SIC 6282, broker-dealers 6211, holding companies 6770)
    where industrial XBRL templates partially apply but distress metrics are unreliable.
    """
    if sic_code is None:
        return 'industrial'
    try:
        s = int(sic_code)
    except (TypeError, ValueError):
        return 'industrial'
    # REITs (most specific — check first)
    if s == 6798:
        return 'reit'
    # Banks
    if 6020 <= s <= 6099 or s in (6710, 6711, 6712):
        return 'bank'
    # Insurance carriers + brokers/agents
    if 6300 <= s <= 6411:
        return 'insurance'
    # Other financial: asset managers (6282), broker-dealers (6211), security exchanges (6231),
    # personal credit institutions (6141), holding companies (6770).
    # These are tagged 'other' so the UI shows a "limited sector support" notice
    # but core fundamentals still render.
    if s in (6141, 6199, 6200, 6211, 6231, 6282, 6770) or 6500 <= s <= 6519:
        return 'other'
    # Batch 7h.22: Biotech/pharma (2834 pharm prep, 2835 in-vitro/in-vivo diagnostics,
    # 2836 biological products, 8731 commercial physical & biological research).
    # Pre-revenue and clinical-stage biotechs have negative or near-zero gross margins
    # and operating losses by design (R&D burn before product approval). Altman Z
    # structurally flags them as distressed even when cash-rich; Beneish ratios are
    # unreliable because they assume a steady-state revenue/cost relationship. Route
    # through 'biotech' bucket which uses the same suppression treatment as 'other'.
    if s in (2834, 2835, 2836, 8731):
        return 'biotech'
    # Everything else is industrial (incl. tech, retail, manufacturing, energy, healthcare, telecom)
    return 'industrial'


def get_company_sic(submissions):
    """Pull SIC code from the EDGAR submissions payload.

    NOTE: SIC code lives in the /submissions/CIK{cik}.json response, NOT in
    /api/xbrl/companyfacts/. The companyfacts endpoint returns only
    {cik, entityName, facts:{us-gaap:{...}, dei:{...}}} — no sector metadata.
    Prior versions of this function read from `facts` and always returned
    None, silently degrading every ticker to the industrial sector bucket
    regardless of actual industry (banks getting industrial-template metrics,
    REITs getting industrial Altman, etc). Fixed in Batch 7h.20.
    """
    if not submissions:
        return None
    sic = submissions.get('sic') or submissions.get('sicCode')
    if sic is None:
        return None
    try:
        return int(sic)
    except (TypeError, ValueError):
        return None


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


def get_submissions(cik: str):
    """Fetch the EDGAR submissions payload for company metadata.

    The submissions endpoint contains sic, sicDescription, name, tickers,
    sector-relevant metadata, and recent filings. Distinct from companyfacts
    (which carries the XBRL numerical series but no sector info).
    Added Batch 7h.20 to fix silent sector-bucket misclassification.
    """
    try:
        r = requests.get(
            f'https://data.sec.gov/submissions/CIK{cik}.json',
            headers=HEADERS, timeout=30
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

FLOW_KEYS = {
    'revenue', 'gross_profit', 'cost_of_revenue', 'operating_income', 'net_income',
    'interest_expense', 'da', 'income_tax',
    'cfo', 'capex', 'cfi', 'cff',
    # Bank flows added in Batch 7h.9
    'interest_income', 'interest_expense_bank', 'net_interest_income',
    'noninterest_income', 'noninterest_expense', 'provision_loan_losses',
    # Insurance flows
    'premiums_earned', 'losses_incurred', 'underwriting_expenses',
    'investment_income',
    # REIT flows
    'rental_income', 'real_estate_revenue',
}

def _is_quarterly_period(entry):
    """True if entry looks like a quarterly (~90 day) period.
    Filters out annual (FY) and YTD (cumulative) entries that share end dates
    with quarterlies but represent different durations.
    """
    start = entry.get('start')
    end = entry.get('end')
    if not start or not end:
        return False
    try:
        from datetime import datetime
        s = datetime.strptime(start, '%Y-%m-%d')
        e = datetime.strptime(end, '%Y-%m-%d')
        days = (e - s).days
    except Exception:
        return False
    if 80 <= days <= 100:
        return True
    return False

def _is_annual_period(entry):
    """True if entry looks like an annual (~365 day) period.
    Used for Beneish M-Score and other forensic metrics that are canonically
    computed on FY data, not TTM. Filters out quarterly (~90d), half-year
    (~180d), and 3Q YTD (~270d) entries.
    """
    start = entry.get('start')
    end = entry.get('end')
    if not start or not end:
        return False
    try:
        from datetime import datetime
        s = datetime.strptime(start, '%Y-%m-%d')
        e = datetime.strptime(end, '%Y-%m-%d')
        days = (e - s).days
    except Exception:
        return False
    if 350 <= days <= 380:
        return True
    return False

def extract_series(facts, key, n=20):
    if not facts or 'us-gaap' not in facts.get('facts', {}):
        return pd.Series(dtype=float)
    usgaap = facts['facts']['us-gaap']
    is_flow = key in FLOW_KEYS
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
            # Batch 7h.8: filter flow series to quarterly (~90d) periods only
            if is_flow:
                qe = [e for e in entries if _is_quarterly_period(e)]
                if qe:
                    entries = qe
            seen = {}
            for e in entries:
                k = e['end']
                if k not in seen or e.get('filed', '') > seen[k].get('filed', ''):
                    seen[k] = e
            sorted_e = sorted(seen.values(), key=lambda x: x['end'])[-n:]
            if len(sorted_e) < 4:
                continue
            div = 1e6 if unit == 'USD' else 1
            s = pd.Series(
                [e['val'] / div for e in sorted_e],
                index=pd.to_datetime([e['end'] for e in sorted_e])
            )
            return s[~s.index.duplicated(keep='last')]
    return pd.Series(dtype=float)

def extract_series_annual(facts, key, n=10):
    """Extract annual (FY) values for a given key. Used by Beneish M-Score
    and other forensic metrics that are canonically computed on FY data.
    Returns up to `n` most recent annual observations.

    For flow keys (revenue, NI, SGA, etc.): filters to ~365d periods.
    For stock keys (assets, receivables, etc.): takes the year-end value
    closest to fiscal year-end. Most issuers report stock balances at
    multiple period ends; we use end-of-fiscal-year matching by selecting
    the value reported with each annual flow.
    """
    if not facts or 'us-gaap' not in facts.get('facts', {}):
        return pd.Series(dtype=float)
    usgaap = facts['facts']['us-gaap']
    is_flow = key in FLOW_KEYS
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
            if is_flow:
                # For flows, filter to ~365d periods (annual)
                ae = [e for e in entries if _is_annual_period(e)]
                if not ae:
                    continue
                entries = ae
            else:
                # For stocks (balance sheet), take period-end values with
                # 'fp' == 'FY' if available. Fall back to end dates that
                # appear to be fiscal year-ends (Dec 31, Sep 30 etc.).
                fy_entries = [e for e in entries if e.get('fp') == 'FY']
                if fy_entries:
                    entries = fy_entries
            seen = {}
            for e in entries:
                k = e['end']
                if k not in seen or e.get('filed', '') > seen[k].get('filed', ''):
                    seen[k] = e
            sorted_e = sorted(seen.values(), key=lambda x: x['end'])[-n:]
            if len(sorted_e) < 2:
                continue
            div = 1e6 if unit == 'USD' else 1
            s = pd.Series(
                [e['val'] / div for e in sorted_e],
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
def quarters_to_trading_days(quarters):
    """Translate the frontend's 'Rolling quarters' slider value to trading days
    for price-based variance computation.

    Approximately 63 trading days per fiscal quarter (252/4). A floor of 180
    trading days is enforced so rolling-90 variance still has headroom for the
    trend regression at small slider values.

    Batch 7h.16: this function lets the user-facing slider actually control
    the ISC variance window. Previously window_days was hardcoded to 252
    everywhere, making the slider non-functional for the headline ISC number.
    """
    try:
        q = int(quarters) if quarters is not None else 12
    except (TypeError, ValueError):
        q = 12
    q = max(4, min(20, q))
    return max(180, q * 63)

def compute_series_trajectory(series, window=6):
    """
    Analyze a quarterly fundamental series without computing the deprecated
    coupling C correlation. Returns trajectory metrics that ARE statistically
    meaningful on small quarterly samples.

    The `window` parameter limits analysis to the last N quarters of the series
    (default 6 = ~18 months). A smaller window emphasizes recent trajectory; a
    larger window captures longer-term direction. Minimum 4 quarters required
    for a valid trend.

    Returns:
      {
        'latest':      most recent observed value,
        'trend_rho':   Spearman correlation with time (-1 to +1),
        'pct_change':  total percent change first to last,
        'avg':         mean across the series,
        'n':           number of observations used (capped at window),
        'direction':   'rising' | 'falling' | 'flat' (based on rho),
        'window':      window quarters parameter used,
      }
    Returns None if insufficient data.
    """
    s = series.dropna() if series is not None else pd.Series()
    if len(s) < 4:
        return None

    # Batch 7h.15: actually use the window parameter — limit trajectory analysis
    # to the last N quarters so the frontend slider has a real effect.
    try:
        w = int(window) if window is not None else 6
        w = max(4, w)  # need at least 4 for a meaningful trend
    except (TypeError, ValueError):
        w = 6
    s = s.iloc[-w:]

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
        'window':     w,
    }

# ── Market data ────────────────────────────────────────────────────────────────
def get_market_data(ticker: str):
    """
    Fetch market data with multi-source fallback chain:
      1. Daily prices: Tiingo → Stooq (via price_data.fetch_daily_prices)
      2. Basic metadata: Tiingo (via price_data.fetch_basic_market_metadata)
      3. Bonus data (options IV, fund metrics): yfinance — accepts failure
    
    Returns the same dict shape as before. Fields not available from any source
    return None and downstream code handles missing values gracefully.
    """
    out = {
        'price': None, 'high_52w': None, 'low_52w': None, 'pct_52w': None,
        'market_cap': None, 'market_cap_bn': None, 'market_cap_m': None,
        'pe': None, 'pb': None, 'ev_ebitda': None, 'beta': None, 'rsi': None,
        'sector': None, 'industry': None, 'company_name': ticker,
        'description': '', 'price_history': [], 'full_price_series': None,
        'atm_iv': None, 'otm_iv': None, 'iv_skew': None,
        'total_revenue': None, 'ebitda': None, 'total_debt': None,
        'free_cashflow': None, 'operating_cashflow': None,
        'current_ratio': None, 'quick_ratio': None, 'debt_to_equity': None,
        'roe': None, 'roa': None, 'gross_margins': None, 'op_margins': None,
        'profit_margins': None, 'revenue_growth': None, 'earnings_growth': None,
        'shares_outstanding': None, 'dividend_yield': None,
        'eps_trailing': None, 'book_value': None,
        'price_source': 'failed',
    }
    
    # 1. Daily prices via Tiingo→Stooq fallback chain
    full_prices, price_source = fetch_daily_prices(ticker, days=1825)
    out['price_source'] = price_source
    
    if full_prices is not None and len(full_prices) > 0:
        out['full_price_series'] = full_prices
        # Tail 60 for chart
        try:
            tail = full_prices.tail(60).tolist()
            out['price_history'] = [round(float(v), 2) for v in tail if v is not None and not np.isnan(v)]
        except Exception:
            out['price_history'] = []
        
        # Latest price
        try:
            out['price'] = round(float(full_prices.iloc[-1]), 2)
        except Exception:
            pass
        
        # 52-week high/low from price series (last 252 trading days)
        try:
            window = full_prices.tail(252)
            out['high_52w'] = round(float(window.max()), 2)
            out['low_52w'] = round(float(window.min()), 2)
            if out['high_52w'] != out['low_52w'] and out['price'] is not None:
                out['pct_52w'] = round(((out['price'] - out['low_52w']) / (out['high_52w'] - out['low_52w'])) * 100, 1)
        except Exception:
            pass
        
        # RSI from full series
        try:
            if len(full_prices) >= 15:
                delta = full_prices.diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi_val = 100 - 100 / (1 + rs.iloc[-1])
                if not (np.isnan(rsi_val) or np.isinf(rsi_val)):
                    out['rsi'] = round(float(rsi_val), 1)
        except Exception:
            pass
    
    # 2. Tiingo metadata (company name, description) — overrides defaults if available
    try:
        meta = fetch_basic_market_metadata(ticker)
        if meta.get('company_name'):
            out['company_name'] = meta['company_name']
        if meta.get('description'):
            out['description'] = meta['description']
        # Tiingo basic metadata doesn't include sector/industry/marketCap (paid plan)
    except Exception:
        pass
    
    # 3. Bonus data via yfinance — accepts failure entirely
    if HAS_YF:
        try:
            tk = yf.Ticker(ticker)
            info = tk.info
            
            def safe_round(v, d=2):
                try:
                    f = float(v)
                    return round(f, d) if not np.isnan(f) and not np.isinf(f) else None
                except:
                    return None
            
            # Only overwrite fields we don't already have from primary sources
            if not out.get('sector'):
                out['sector'] = info.get('sector')
            if not out.get('industry'):
                out['industry'] = info.get('industry')
            if not out.get('description'):
                desc = info.get('longBusinessSummary') or ''
                out['description'] = desc[:300] if desc else ''
            
            mktcap = info.get('marketCap')
            if mktcap:
                out['market_cap'] = mktcap
                out['market_cap_bn'] = safe_round(mktcap/1e9, 2)
                out['market_cap_m'] = safe_round(mktcap/1e6, 1)
            
            out['pe'] = safe_round(info.get('trailingPE'), 1)
            out['pb'] = safe_round(info.get('priceToBook'), 2)
            out['ev_ebitda'] = safe_round(info.get('enterpriseToEbitda'), 1)
            out['beta'] = safe_round(info.get('beta'), 2)
            out['total_revenue'] = safe_round(info.get('totalRevenue'), 0)
            out['ebitda'] = safe_round(info.get('ebitda'), 0)
            out['total_debt'] = safe_round(info.get('totalDebt'), 0)
            out['free_cashflow'] = safe_round(info.get('freeCashflow'), 0)
            out['operating_cashflow'] = safe_round(info.get('operatingCashflow'), 0)
            out['current_ratio'] = safe_round(info.get('currentRatio'), 2)
            out['quick_ratio'] = safe_round(info.get('quickRatio'), 2)
            out['debt_to_equity'] = safe_round(info.get('debtToEquity'), 2)
            out['roe'] = safe_round(info.get('returnOnEquity'), 4)
            out['roa'] = safe_round(info.get('returnOnAssets'), 4)
            out['gross_margins'] = safe_round(info.get('grossMargins'), 4)
            out['op_margins'] = safe_round(info.get('operatingMargins'), 4)
            out['profit_margins'] = safe_round(info.get('profitMargins'), 4)
            out['revenue_growth'] = safe_round(info.get('revenueGrowth'), 4)
            out['earnings_growth'] = safe_round(info.get('earningsGrowth'), 4)
            out['shares_outstanding'] = info.get('sharesOutstanding')
            out['dividend_yield'] = safe_round(info.get('dividendYield'), 4)
            out['eps_trailing'] = safe_round(info.get('trailingEps'), 2)
            out['book_value'] = safe_round(info.get('bookValue'), 2)
            
            # Override company name if Tiingo didn't get it
            if not out.get('company_name') or out['company_name'] == ticker:
                out['company_name'] = info.get('longName') or info.get('shortName') or ticker
            
            # Options IV
            try:
                expiries = tk.options
                price = out.get('price')
                if expiries and price:
                    chain = tk.option_chain(expiries[0])
                    calls = chain.calls.copy()
                    puts = chain.puts.copy()
                    if not calls.empty:
                        calls['dist'] = abs(calls['strike'] - price)
                        iv = calls.loc[calls['dist'].idxmin(), 'impliedVolatility']
                        out['atm_iv'] = round(float(iv)*100, 1) if not np.isnan(float(iv)) else None
                    if not puts.empty:
                        puts['dist'] = abs(puts['strike'] - price*0.90)
                        iv2 = puts.loc[puts['dist'].idxmin(), 'impliedVolatility']
                        out['otm_iv'] = round(float(iv2)*100, 1) if not np.isnan(float(iv2)) else None
                    if out['atm_iv'] and out['otm_iv']:
                        out['iv_skew'] = round(out['otm_iv'] - out['atm_iv'], 1)
            except Exception:
                pass
        except Exception:
            # yfinance failure is fine — we already have prices from Tiingo/Stooq
            pass
    
    return out

# ── Traditional metrics ────────────────────────────────────────────────────────
def _build_price_only_response(ticker: str, mkt: dict, window: int):
    """Batch 7h.14: minimal response for stocks-only analysis mode.

    Skips EDGAR lookup entirely. Returns variance EWS, price history,
    and ticker metadata. Fundamental panels (income statement, balance sheet,
    cash flow, traditional metrics, quarterly history) are intentionally
    omitted with a clear flag so the frontend can show appropriate fallbacks.
    """
    # Batch 7h.16: thread the user-selected window through to variance
    # computation. The frontend slider sends `window` in quarters; translate
    # to trading days so compute_variance_score actually respects the user's
    # selection. Previously window_days was hardcoded to 252.
    variance_window_days = quarters_to_trading_days(window)
    variance_score_pr = None
    full_series = mkt.get('full_price_series')
    if full_series is not None and len(full_series) >= 90:
        variance_score_pr = compute_variance_score(
            full_series,
            window_days=variance_window_days,
            rolling_window=90,
        )

    if variance_score_pr and 'error' not in variance_score_pr:
        primary_variance = variance_score_pr.get('mean_variance')
        primary_trend    = variance_score_pr.get('variance_trend')
        primary_ratio    = variance_score_pr.get('variance_ratio')
        primary_regime   = variance_score_pr.get('regime')
        score_available  = True
        score_error      = None
    else:
        primary_variance = None
        primary_trend    = None
        primary_ratio    = None
        primary_regime   = 'unavailable'
        score_available  = False
        score_error      = (variance_score_pr.get('error') if variance_score_pr else 'no_price_data')

    # Resilient sector text from mkt
    sector_text = mkt.get('sector') or mkt.get('industry') or ''

    response = {
        'ticker': ticker,
        'company_name': mkt.get('company_name') or ticker,
        'analysis_mode': 'price-only',
        'price_only_notice': (
            'Stocks-only analysis. EDGAR fundamentals (income statement, balance sheet, '
            'cash flow, traditional metrics, peer comparison) are not loaded in this mode. '
            'Switch to EDGAR mode for full fundamental analysis where filings are available.'
        ),
        'sector': sector_text,
        'industry': mkt.get('industry') or '',
        'sector_bucket': None,        # not classified without SIC
        'sic_code': None,
        'sic_description': None,
        'description': mkt.get('description') or '',
        'analysis_date': str(pd.Timestamp.now().date()),
        'data_sources': {
            'edgar': False,
            'yfinance': bool(mkt.get('price')),
            'prices': mkt.get('price_source', 'failed'),
        },
        'window': window,

        # Market data — Batch 7h.15: align field names with EDGAR-mode response
        # so the frontend can render the same market panel in either mode.
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

        # Variance EWS — primary signal in this mode
        'isc': {
            'C':                 primary_variance,
            'regime':            primary_regime,
            'available':         score_available,
            'error':             score_error,
            'variance_score':    primary_variance,
            'variance_trend':    primary_trend,
            'variance_ratio':    primary_ratio,
            'plain':             None,   # frontend will compose from regime
            'stability_score':   None,
        },

        # Explicitly empty fundamental blocks — frontend uses these flags to
        # render "Not loaded in price-only mode" placeholders rather than
        # silently degrading to zero values.
        'income_statement': None,
        'balance_sheet':    None,
        'cash_flow':        None,
        'traditional':      None,
        'quarterly_history': None,
        'sector_metrics':   None,

        # Series trajectories also unavailable
        'series_trajectories': {},

        # News passthrough — empty in price-only mode for now (Tiingo news lookup
        # is part of the full EDGAR path; price-only is intentionally minimal)
        'news': [],
    }
    return response


def _build_quarterly_history(raw, n_quarters=12):
    """Batch 7h.10: Extract quarterly time series for the frontend to render trend lines.

    Returns a dict keyed by metric name; each value is a list of {date, val} dicts,
    sorted oldest-to-newest, limited to the last n_quarters entries.

    Includes all flow series (revenue, operating_income, etc.) where quarterly
    values are meaningful. Skips stock series (balance sheet items) that are
    point-in-time and don't trend the same way.
    """
    METRICS = [
        'revenue', 'gross_profit', 'operating_income', 'net_income',
        'cfo', 'capex',
        # Sector-specific
        'net_interest_income', 'noninterest_income', 'noninterest_expense',
        'premiums_earned', 'losses_incurred',
        'rental_income',
    ]
    out = {}
    for key in METRICS:
        s = raw.get(key)
        if s is None or len(s) == 0:
            continue
        s = s.dropna()
        if len(s) == 0:
            continue
        tail = s.iloc[-n_quarters:]
        out[key] = [
            {'date': str(idx.date()), 'val': float(round(val, 2))}
            for idx, val in zip(tail.index, tail.values)
        ]
    return out


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

def aligned_ttm_ratio(numerator_series, denominator_series):
    """Batch 7h.15: Compute a TTM ratio only when numerator and denominator
    come from the same fiscal quarters.

    The plain `ttm(num) / ttm(den)` approach can divide a TTM revenue ending Q1
    against a TTM operating income ending Q4 of the prior year if the two
    series have different latest-available filings. That produces nonsensical
    ratios like operating margin > gross margin (AMZN) or net margin > operating
    margin (GME).

    This function:
      1. Finds the last 4 quarters where BOTH series have observations on the
         same end-date.
      2. Sums each over those aligned quarters.
      3. Returns the ratio, or None if alignment fails or denominator is zero.
    """
    if numerator_series is None or denominator_series is None:
        return None
    n = numerator_series.dropna() if hasattr(numerator_series, 'dropna') else None
    d = denominator_series.dropna() if hasattr(denominator_series, 'dropna') else None
    if n is None or d is None or len(n) == 0 or len(d) == 0:
        return None
    common = n.index.intersection(d.index)
    if len(common) < 4:
        return None
    aligned_n = n.loc[common].iloc[-4:]
    aligned_d = d.loc[common].iloc[-4:]
    if len(aligned_n) < 4 or len(aligned_d) < 4:
        return None
    num_sum = float(aligned_n.sum())
    den_sum = float(aligned_d.sum())
    if den_sum == 0 or np.isnan(den_sum) or np.isinf(den_sum):
        return None
    if np.isnan(num_sum) or np.isinf(num_sum):
        return None
    v = num_sum / den_sum
    if np.isnan(v) or np.isinf(v):
        return None
    return round(v, 3)

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

# ── Beneish M-Score (forensic accounting fraud probability) ───────────────────
# Reference: Beneish, M. D. (1999). "The Detection of Earnings Manipulation."
#   Financial Analysts Journal 55(5), 24-36.
# Original sample: 74 manipulators vs 2,332 non-manipulators, 1982-1992.
# Reported test accuracy: ~76% true positives, ~17.5% false positives at M > -1.78.
#
# Eight ratios capturing year-over-year changes that are characteristic of
# earnings manipulation:
#
#   DSRI = (Receivables_t / Revenue_t) / (Receivables_t-1 / Revenue_t-1)
#     Days sales in receivables. Aggressive revenue recognition pushes this up.
#
#   GMI  = Gross_Margin_t-1 / Gross_Margin_t
#     Gross margin index. >1 means margin deteriorated (motive to manipulate).
#
#   AQI  = (1 - (CA_t + PPE_t) / TA_t) / (1 - (CA_t-1 + PPE_t-1) / TA_t-1)
#     Asset quality index. Rising "other assets" share → soft assets, capitalized expenses.
#
#   SGI  = Revenue_t / Revenue_t-1
#     Sales growth index. High growth firms have stronger manipulation incentives.
#
#   DEPI = (Dep_t-1 / (Dep_t-1 + GrossPPE_t-1)) / (Dep_t / (Dep_t + GrossPPE_t))
#     Depreciation index. >1 means depreciation rate slowed (income inflation).
#
#   SGAI = (SGA_t / Rev_t) / (SGA_t-1 / Rev_t-1)
#     SG&A index. Disproportionate SG&A growth signals operating efficiency decline.
#
#   TATA = (NI_t - CFO_t) / TA_t
#     Total accruals / total assets. Higher accruals → more discretionary accounting.
#
#   LVGI = ((LTD_t + CL_t) / TA_t) / ((LTD_t-1 + CL_t-1) / TA_t-1)
#     Leverage index. Rising leverage may motivate covenant manipulation.
#
# M = -4.84 + 0.92·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI
#     + 0.115·DEPI - 0.172·SGAI + 4.679·TATA - 0.327·LVGI
#
# Thresholds:
#   M > -1.78 (original 1999): elevated probability of manipulation. ~76% sensitivity.
#   M > -2.22 (more recent calibration, e.g. Beneish/Lee/Nichols 2013): stricter,
#            higher specificity, reduced false positives.
#
# This is a SCREENING tool, not a verdict. A high M-Score is a flag for closer
# investigation. Banks, insurers, and utilities don't fit the model (different
# accounting structure) — caller should suppress for those sectors.

def _safe_ratio(num, den):
    """Return num/den or None if either is None or den is zero/near-zero."""
    if num is None or den is None:
        return None
    try:
        if abs(den) < 1e-9:
            return None
        return num / den
    except (TypeError, ZeroDivisionError):
        return None

def compute_beneish_m_score(rev_t, rev_p, recv_t, recv_p, gp_t, gp_p,
                             ca_t, ca_p, ppe_t, ppe_p, ta_t, ta_p,
                             dep_t, dep_p, gross_ppe_t, gross_ppe_p,
                             sga_t, sga_p, ni_t, cfo_t,
                             ltd_t, ltd_p, cl_t, cl_p):
    """Compute Beneish M-Score and its 8 components.

    All inputs are scalar values for two consecutive periods (t = current,
    p = prior). Returns a dict with each ratio, the M-Score, threshold flags,
    and a list of "missing inputs" that prevented full computation.

    Returns None if essential inputs are missing for the core ratios.
    """
    missing = []

    # DSRI: Days sales in receivables index
    if all(v is not None for v in [recv_t, rev_t, recv_p, rev_p]) and rev_t and rev_p:
        dsri_num = _safe_ratio(recv_t, rev_t)
        dsri_den = _safe_ratio(recv_p, rev_p)
        dsri = _safe_ratio(dsri_num, dsri_den)
    else:
        dsri = None
        missing.append('DSRI (receivables/revenue)')

    # GMI: Gross margin index (prior / current; >1 = deterioration)
    # Batch 7h.22: guard against nonsensical sign when one gross margin is
    # negative or both yield a non-positive ratio. The ratio is only meaningful
    # for two same-sign positive margins. Examples of the original bug: META
    # was returning GMI = -0.829 due to bad intermediate values (likely a stale
    # cost_of_revenue XBRL value flowing through the fallback path).
    gm_t = gm_p = None
    if all(v is not None for v in [gp_t, gp_p, rev_t, rev_p]) and rev_t and rev_p:
        gm_t = _safe_ratio(gp_t, rev_t)
        gm_p = _safe_ratio(gp_p, rev_p)
        # Both margins must be positive for the ratio to be interpretable.
        # If either is non-positive (negative or zero), the firm has cost-side
        # accounting structure that breaks the Beneish assumption.
        if gm_t is not None and gm_p is not None and gm_t > 0 and gm_p > 0:
            gmi = gm_p / gm_t
        else:
            gmi = None
            missing.append('GMI (one or both gross margins non-positive: gm_t={}, gm_p={})'.format(
                round(gm_t, 4) if gm_t is not None else None,
                round(gm_p, 4) if gm_p is not None else None))
    else:
        gmi = None
        missing.append('GMI (gross margin)')

    # AQI: Asset quality index — share of "other" non-current assets
    # (1 - (CA + PPE)/TA) is the "other assets" fraction.
    if all(v is not None for v in [ca_t, ppe_t, ta_t, ca_p, ppe_p, ta_p]) and ta_t and ta_p:
        oa_t = 1 - _safe_ratio(ca_t + ppe_t, ta_t)
        oa_p = 1 - _safe_ratio(ca_p + ppe_p, ta_p)
        # Avoid division by tiny/zero "other assets" denominators (mature firms)
        if oa_p is not None and abs(oa_p) > 0.001:
            aqi = _safe_ratio(oa_t, oa_p)
        else:
            aqi = 1.0  # neutral when prior other-assets is negligible
    else:
        aqi = None
        missing.append('AQI (asset quality)')

    # SGI: Sales growth index
    if rev_t is not None and rev_p is not None and rev_p:
        sgi = _safe_ratio(rev_t, rev_p)
    else:
        sgi = None
        missing.append('SGI (sales growth)')

    # DEPI: Depreciation rate index
    # rate_t = dep_t / (dep_t + gross_ppe_t)
    # rate_p = dep_p / (dep_p + gross_ppe_p)
    # DEPI = rate_p / rate_t  (>1 means rate slowed)
    if all(v is not None for v in [dep_t, dep_p, gross_ppe_t, gross_ppe_p]):
        rate_t = _safe_ratio(dep_t, dep_t + gross_ppe_t)
        rate_p = _safe_ratio(dep_p, dep_p + gross_ppe_p)
        depi = _safe_ratio(rate_p, rate_t)
    else:
        depi = None
        missing.append('DEPI (depreciation rate)')

    # SGAI: SG&A index (higher = inefficiency growing)
    if all(v is not None for v in [sga_t, sga_p, rev_t, rev_p]) and rev_t and rev_p:
        sgai_num = _safe_ratio(sga_t, rev_t)
        sgai_den = _safe_ratio(sga_p, rev_p)
        sgai = _safe_ratio(sgai_num, sgai_den)
    else:
        sgai = None
        missing.append('SGAI (SG&A)')

    # TATA: Total accruals / total assets (current period only)
    if all(v is not None for v in [ni_t, cfo_t, ta_t]) and ta_t:
        tata = _safe_ratio(ni_t - cfo_t, ta_t)
    else:
        tata = None
        missing.append('TATA (accruals)')

    # LVGI: Leverage index
    if all(v is not None for v in [ltd_t, cl_t, ta_t, ltd_p, cl_p, ta_p]) and ta_t and ta_p:
        lev_t = _safe_ratio(ltd_t + cl_t, ta_t)
        lev_p = _safe_ratio(ltd_p + cl_p, ta_p)
        lvgi = _safe_ratio(lev_t, lev_p)
    else:
        lvgi = None
        missing.append('LVGI (leverage)')

    # Compute M-Score: require at least 5 of 8 ratios to attempt
    available = [x for x in [dsri, gmi, aqi, sgi, depi, sgai, tata, lvgi] if x is not None]
    if len(available) < 5:
        return {
            'm_score': None,
            'components': {
                'DSRI': dsri, 'GMI': gmi, 'AQI': aqi, 'SGI': sgi,
                'DEPI': depi, 'SGAI': sgai, 'TATA': tata, 'LVGI': lvgi,
            },
            'missing': missing,
            'n_ratios': len(available),
            'flag_original': None,
            'flag_strict': None,
            'note': f'Insufficient data: only {len(available)} of 8 ratios computable',
            'diagnostic': {
                'rev_t': rev_t, 'rev_p': rev_p,
                'gp_t': gp_t, 'gp_p': gp_p,
                'gm_t': round(gm_t, 4) if gm_t is not None else None,
                'gm_p': round(gm_p, 4) if gm_p is not None else None,
            },
        }

    # Beneish 1999 coefficients
    # When a ratio is missing, substitute with the "neutral" value (1.0 for
    # ratios, 0 for TATA) and flag in the response. This is a conservative
    # approach — it neither inflates nor deflates the score for missing data.
    m = (-4.84
         + 0.92  * (dsri if dsri is not None else 1.0)
         + 0.528 * (gmi  if gmi  is not None else 1.0)
         + 0.404 * (aqi  if aqi  is not None else 1.0)
         + 0.892 * (sgi  if sgi  is not None else 1.0)
         + 0.115 * (depi if depi is not None else 1.0)
         - 0.172 * (sgai if sgai is not None else 1.0)
         + 4.679 * (tata if tata is not None else 0.0)
         - 0.327 * (lvgi if lvgi is not None else 1.0))

    # Identify top contributors by absolute contribution to score
    contributions = {}
    if dsri is not None: contributions['DSRI'] = 0.92 * dsri
    if gmi  is not None: contributions['GMI']  = 0.528 * gmi
    if aqi  is not None: contributions['AQI']  = 0.404 * aqi
    if sgi  is not None: contributions['SGI']  = 0.892 * sgi
    if depi is not None: contributions['DEPI'] = 0.115 * depi
    if sgai is not None: contributions['SGAI'] = -0.172 * sgai
    if tata is not None: contributions['TATA'] = 4.679 * tata
    if lvgi is not None: contributions['LVGI'] = -0.327 * lvgi

    return {
        'm_score': round(m, 3),
        'components': {
            'DSRI': round(dsri, 3) if dsri is not None else None,
            'GMI':  round(gmi, 3)  if gmi  is not None else None,
            'AQI':  round(aqi, 3)  if aqi  is not None else None,
            'SGI':  round(sgi, 3)  if sgi  is not None else None,
            'DEPI': round(depi, 3) if depi is not None else None,
            'SGAI': round(sgai, 3) if sgai is not None else None,
            'TATA': round(tata, 4) if tata is not None else None,
            'LVGI': round(lvgi, 3) if lvgi is not None else None,
        },
        'contributions': {k: round(v, 3) for k, v in contributions.items()},
        'missing': missing,
        'n_ratios': len(available),
        'flag_original':  m > -1.78,   # 1999 threshold (more sensitive)
        'flag_strict':    m > -2.22,   # later calibration (stricter, fewer false positives)
        'note': None if not missing else f'Computed with {len(available)} of 8 ratios; missing: {", ".join(missing)}',
        # Batch 7h.22: intermediate values for diagnostic inspection. Helps trace
        # cases where a ratio is None or returns an unexpected value — frontend
        # can show these in a "data quality" expandable section for power users.
        'diagnostic': {
            'rev_t': rev_t, 'rev_p': rev_p,
            'gp_t': gp_t, 'gp_p': gp_p,
            'gm_t': round(gm_t, 4) if gm_t is not None else None,
            'gm_p': round(gm_p, 4) if gm_p is not None else None,
        },
    }


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
async def analyze(ticker: str, window: int = 12, mode: str = "edgar"):
    """
    Analyze a ticker.

    mode='edgar' (default): full EDGAR-backed analysis with variance EWS,
        fundamentals, traditional metrics, sector context, peer comparison,
        and quarterly history. Best for US public companies with complete
        XBRL filings.

    mode='price-only': stock-price-derived analysis only. Returns variance EWS,
        price history, and ticker metadata. Fundamental panels (income statement,
        balance sheet, cash flow, traditional metrics, peer comparison) are
        intentionally omitted. Use for foreign filers, very new IPOs, or when
        you want a faster price-only read.
    """
    ticker = ticker.upper().strip()
    if mode not in ("edgar", "price-only"):
        mode = "edgar"

    # 1. Market data (yfinance)
    mkt = get_market_data(ticker)
    if 'error' in mkt and not mkt.get('price'):
        # yfinance failed — try EDGAR only
        pass

    # Batch 7h.14: short-circuit for price-only mode.
    # Skip EDGAR/CIK lookup, skip facts retrieval, skip all derived metrics.
    # Return a minimal response: variance EWS + price history + sector text
    # from yfinance only.
    if mode == "price-only":
        return _build_price_only_response(ticker, mkt, window)

    # 2. EDGAR
    cik = get_cik(ticker)
    facts = get_facts(cik) if cik else None
    submissions = get_submissions(cik) if cik else None

    # Batch 7h.9: detect sector from SIC code
    # Batch 7h.20: SIC lives in submissions endpoint, not companyfacts. Prior
    # version always returned None, defaulting every ticker to industrial bucket.
    sic_code = get_company_sic(submissions)
    sic_description = submissions.get('sicDescription') if submissions else None
    sector_bucket = detect_sector(sic_code)

    # 3. Extract series from EDGAR
    raw = {}
    if facts:
        for key in TAG_MAP:
            raw[key] = extract_series(facts, key)

    # 3b. Extract annual (FY) series for Beneish M-Score and other forensic metrics
    # Only for keys actually used; saves fetcher iterations.
    BENEISH_KEYS = [
        'revenue', 'gross_profit', 'cost_of_revenue', 'net_income',
        'cfo', 'sga', 'da', 'gross_ppe', 'accumulated_depreciation',
        'total_assets', 'current_assets', 'ppe_net',
        'long_term_debt', 'current_liabilities', 'receivables',
    ]
    raw_annual = {}
    if facts:
        for key in BENEISH_KEYS:
            raw_annual[key] = extract_series_annual(facts, key)

    # 4. Derive TTM values
    # Batch 7h.17 (Tesla fix): keep rev_ttm fallback chain conservative.
    # Path 1: full TTM from EDGAR quarterly revenue (4+ quarters required)
    # Path 2: yfinance totalRevenue (TTM, in dollars; convert to millions)
    # Note: deliberately do NOT annualize from a single quarter — for seasonal
    # businesses (retail, agricultural) this would produce wildly wrong numbers.
    # Better to surface N/A than a fabricated annualization.
    rev_ttm   = ttm(raw.get('revenue'))   or (mkt.get('total_revenue', 0) / 1e6 if mkt.get('total_revenue') else None)
    gp_ttm    = ttm(raw.get('gross_profit'))
    oi_ttm_for_check = ttm(raw.get('operating_income'))  # needed for sanity check below
    # Batch 7h.16: track provenance — only the cost_of_revenue fallback path produces
    # a gp_ttm where safe_div(gp_ttm, rev_ttm) is safe (because both numerator and
    # denominator come from the same TTM window). When gp_ttm comes from the raw
    # gross_profit series, period misalignment with revenue is possible and we must
    # require aligned_ttm_ratio.
    gp_ttm_from_fallback = False
    # Batch 7h.22: accounting-identity sanity check. By definition, operating income
    # (= gross profit - operating expenses) must be <= gross profit. If gp_ttm is
    # smaller than oi_ttm, the XBRL extraction picked up a segment-only or wrong-context
    # value (observed for AMZN where gp_ttm came back as ~$4B vs real ~$329B, but
    # oi_ttm came back as the correct ~$79B). Reject and trigger the cost-of-revenue
    # fallback chain. Threshold: require gp_ttm >= 0.9 * oi_ttm to allow for small
    # rounding/timing differences while catching the segment-misextraction case.
    gp_ttm_sanity_failed = False
    if gp_ttm is not None and oi_ttm_for_check is not None and oi_ttm_for_check > 0:
        if gp_ttm < 0.9 * oi_ttm_for_check:
            gp_ttm_sanity_failed = True
            gp_ttm = None  # treat as missing so fallback fires
    # Batch 7h.15: Fallback for services companies, REITs, banks that don't report
    # GrossProfit directly. Compute from revenue minus cost of revenue when available.
    if gp_ttm is None:
        cor_ttm = ttm(raw.get('cost_of_revenue'))
        if rev_ttm is not None and cor_ttm is not None:
            gp_ttm = round(rev_ttm - cor_ttm, 2)
            gp_ttm_from_fallback = True
    oi_ttm    = ttm(raw.get('operating_income'))
    ni_ttm    = ttm(raw.get('net_income'))
    int_ttm   = ttm(raw.get('interest_expense'))
    tax_ttm   = ttm(raw.get('income_tax'))   # Batch 7h.22: for effective tax rate in ROIC
    da_ttm    = ttm(raw.get('da'))
    cfo_ttm   = ttm(raw.get('cfo'))       or (mkt.get('operating_cashflow', 0) / 1e6 if mkt.get('operating_cashflow') else None)
    capex_ttm = ttm(raw.get('capex'))
    cfi_ttm   = ttm(raw.get('cfi'))
    cff_ttm   = ttm(raw.get('cff'))

    # Batch 7h.9: sector-specific TTM derivations
    # Bank
    interest_income_ttm    = ttm(raw.get('interest_income'))
    interest_expense_b_ttm = ttm(raw.get('interest_expense_bank'))
    net_int_income_ttm     = ttm(raw.get('net_interest_income'))
    if not net_int_income_ttm and interest_income_ttm and interest_expense_b_ttm:
        net_int_income_ttm = round(interest_income_ttm - interest_expense_b_ttm, 2)
    noninterest_income_ttm  = ttm(raw.get('noninterest_income'))
    noninterest_expense_ttm = ttm(raw.get('noninterest_expense'))
    provision_ll_ttm        = ttm(raw.get('provision_loan_losses'))
    loans_recv             = last(raw.get('loans_receivable'))
    deposits_total         = last(raw.get('deposits'))
    tier1_cap              = last(raw.get('tier1_capital'))
    rwa                    = last(raw.get('risk_weighted_assets'))
    allowance_ll           = last(raw.get('allowance_loan_losses'))

    # Insurance
    premiums_ttm        = ttm(raw.get('premiums_earned'))
    losses_ttm          = ttm(raw.get('losses_incurred'))
    underwrite_exp_ttm  = ttm(raw.get('underwriting_expenses'))
    inv_income_ttm      = ttm(raw.get('investment_income'))
    investments_held    = last(raw.get('investments_held'))
    reserves_total      = last(raw.get('reserves'))

    # REIT
    rental_income_ttm   = ttm(raw.get('rental_income'))
    real_estate_revenue_ttm = ttm(raw.get('real_estate_revenue'))
    real_estate_at_cost = last(raw.get('real_estate_at_cost'))
    real_estate_acc_dep = last(raw.get('real_estate_accumulated_dep'))
    real_estate_net     = last(raw.get('real_estate_net'))
    mortgages_payable   = last(raw.get('mortgages_payable'))
    # FFO = Net Income + Real-Estate Depreciation - Gains on Sales (we approximate using total D&A if avail)
    ffo_ttm = None
    if ni_ttm is not None and da_ttm is not None:
        ffo_ttm = round(ni_ttm + da_ttm, 2)

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
    ap   = last(raw.get('accounts_payable'))
    ltd  = last(raw.get('long_term_debt'))
    std  = last(raw.get('short_term_debt'))
    tl   = last(raw.get('total_liabilities'))
    te   = last(raw.get('total_equity'))
    # Batch 7h.16: Computational fallback when EDGAR returns total_assets and
    # total_equity but not the standalone Liabilities tag. By the fundamental
    # accounting identity (Assets = Liabilities + Equity), tl = ta - te is
    # always correct when both are populated. This recovers Altman Z on tickers
    # like AMZN where the Liabilities tag is intermittently missing from the
    # EDGAR XBRL response while ta and te both populate.
    if tl is None and ta is not None and te is not None:
        tl = round(ta - te, 2)
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
    # Batch 7h.16: Use aligned_ttm_ratio to avoid period-mismatch bugs where
    # numerator and denominator come from different fiscal quarters (which can
    # produce mathematically impossible results like op_margin > gross_margin
    # for AMZN, or net_margin > op_margin for GME).
    #
    # Fallback policy (changed from 7h.15):
    #   - gross_margin: ONLY fall back to safe_div(gp_ttm, rev_ttm) when gp_ttm
    #     came from the cost_of_revenue derivation path (which already used
    #     period-aligned TTM totals via ttm()). If gp_ttm came from the raw
    #     gross_profit series and aligned_ttm_ratio failed, accept None — bad
    #     numbers are worse than missing numbers.
    #   - op_margin / net_margin: do NOT fall back to safe_div on raw TTM. If
    #     periods don't align, accept None. yfinance's op_margins is also
    #     accepted as a last resort.
    gross_margin  = aligned_ttm_ratio(raw.get('gross_profit'), raw.get('revenue'))
    if gross_margin is None and gp_ttm_from_fallback and gp_ttm is not None and rev_ttm is not None:
        gross_margin = safe_div(gp_ttm, rev_ttm)
    if gross_margin is None and mkt.get('gross_margins'):
        gross_margin = mkt['gross_margins']
    op_margin     = aligned_ttm_ratio(raw.get('operating_income'), raw.get('revenue')) or mkt.get('op_margins')
    net_margin    = aligned_ttm_ratio(raw.get('net_income'), raw.get('revenue'))       or mkt.get('profit_margins')

    # Batch 7h.16: Server-side sanity check. If op_margin > gross_margin by more
    # than half a percentage point of a unit (i.e. op_margin numerically larger),
    # the two values came from misaligned data. Force both to None so the frontend
    # gets clean N/As rather than a contradictory pair plus a warning. This catches
    # the residual AMZN case where one value passes alignment and the other doesn't,
    # or where yfinance's op_margins disagrees with our gross_margin.
    if (gross_margin is not None and op_margin is not None
            and op_margin > gross_margin + 0.005):
        gross_margin = None
        op_margin = None
    # Same check for net > op (the GME pattern)
    if (op_margin is not None and net_margin is not None
            and net_margin > op_margin + 0.05):
        # Less aggressive — net can occasionally exceed op slightly due to one-time
        # tax benefits or non-operating gains, so use a 5pp threshold not 0.5pp.
        # If exceeded, null both rather than show contradictory metrics.
        op_margin = None
        net_margin = None

    fcf_margin    = safe_div(fcf_ttm, rev_ttm)
    roa           = safe_div(ni_ttm, ta)      or mkt.get('roa')
    roe           = safe_div(ni_ttm, te)      or mkt.get('roe')
    curr_ratio    = safe_div(ca, cl)          or mkt.get('current_ratio')
    quick_ratio   = safe_div((ca or 0)-(inv or 0), cl) if (ca and cl) else mkt.get('quick_ratio')
    de_ratio      = safe_div(total_debt, te)  or mkt.get('debt_to_equity')
    debt_ebitda   = safe_div(total_debt, ebitda_ttm)
    int_coverage  = aligned_ttm_ratio(raw.get('operating_income'), raw.get('interest_expense'))
    if int_coverage is None:
        int_coverage = safe_div(oi_ttm, int_ttm)
    asset_turn    = safe_div(rev_ttm, ta)
    cash_conv     = aligned_ttm_ratio(raw.get('cfo'), raw.get('net_income'))
    if cash_conv is None:
        cash_conv = safe_div(cfo_ttm, ni_ttm)

    # ── Phase 2a: FP&A depth metrics ─────────────────────────────────────────
    # These are computed from existing inputs; no new TAG_MAP fields needed.
    # Each metric is paired with the same safety patterns as existing ratios:
    # None when inputs missing, conservative on edge cases.

    # EBITDA Margin = EBITDA / Revenue
    # Different from FCF margin — captures operating profitability before D&A.
    ebitda_margin = safe_div(ebitda_ttm, rev_ttm)

    # FCF Yield = FCF / Market Cap
    # Inverse of P/FCF. Higher = more cash returned per dollar invested.
    # Different from FCF Margin (FCF/Revenue). Both are useful.
    fcf_yield = safe_div(fcf_ttm, mkt.get('market_cap_m'))

    # ROIC = NOPAT / Invested Capital
    # NOPAT approximation: Operating Income * (1 - effective tax rate)
    # Batch 7h.22: compute effective tax rate from actual filings rather than
    # using a 21% federal default. Effective rate = tax_expense / pretax_income,
    # where pretax_income ≈ net_income + income_tax (since pretax = pretax_income,
    # and net = pretax - tax → pretax = net + tax).
    # Fall back to 21% only when tax_ttm or ni_ttm unavailable.
    # Sanity bounds: clamp to [0.0, 0.45] — outside this range likely indicates
    # tax loss carryforwards, deferred tax credits, or anomalous one-time items
    # that don't reflect ongoing operations. The 21% default is more honest in
    # those cases than a wildly off effective rate.
    # Invested Capital = Total Debt + Total Equity (simplified — excludes
    # operating leases and other adjustments analysts sometimes make)
    DEFAULT_TAX_RATE = 0.21
    effective_tax_rate = DEFAULT_TAX_RATE
    effective_tax_rate_source = 'default_21pct'
    if tax_ttm is not None and ni_ttm is not None:
        pretax_ttm = ni_ttm + tax_ttm
        if pretax_ttm and pretax_ttm > 0:  # positive pretax: meaningful effective rate
            raw_rate = tax_ttm / pretax_ttm
            if 0.0 <= raw_rate <= 0.45:
                effective_tax_rate = raw_rate
                effective_tax_rate_source = 'computed_from_filings'
    invested_capital = None
    if total_debt is not None and te is not None:
        invested_capital = total_debt + te
    nopat = None
    if oi_ttm is not None:
        nopat = round(oi_ttm * (1 - effective_tax_rate), 2)
    roic = safe_div(nopat, invested_capital)

    # Working capital cycle metrics — DSO, DIO, DPO
    # These reveal channel stuffing, inventory buildup, and supplier squeeze
    # patterns that don't show in margin analysis alone.
    # Use 365-day basis for canonical interpretation.
    # cor_ttm is computed above only when gp_ttm path triggers; make it
    # available unconditionally for DIO.
    if 'cor_ttm' not in dir() or cor_ttm is None:
        cor_ttm = ttm(raw.get('cost_of_revenue'))

    # DSO (Days Sales Outstanding) = Receivables / Revenue * 365
    # Rising DSO can signal aggressive revenue recognition or weakening
    # credit quality of customers.
    dso = None
    if rec is not None and rev_ttm is not None and rev_ttm > 0:
        dso = round((rec / rev_ttm) * 365, 1)

    # DIO (Days Inventory Outstanding) = Inventory / COGS * 365
    # Rising DIO can signal demand weakening or inventory buildup before
    # a writedown.
    dio = None
    if inv is not None and cor_ttm is not None and cor_ttm > 0:
        dio = round((inv / cor_ttm) * 365, 1)

    # DPO (Days Payable Outstanding) = Accounts Payable / COGS * 365
    # Rising DPO can signal cash flow stress (stretching supplier payments)
    # or, alternatively, improved negotiating leverage. Context matters.
    dpo = None
    if ap is not None and cor_ttm is not None and cor_ttm > 0:
        dpo = round((ap / cor_ttm) * 365, 1)

    # Cash Conversion Cycle = DSO + DIO - DPO
    # Negative is excellent (collect before paying — Amazon model).
    # Positive is normal (cash tied up in operations).
    # Trend matters more than absolute level.
    ccc = None
    if dso is not None and dio is not None and dpo is not None:
        ccc = round(dso + dio - dpo, 1)

    # Goodwill / Total Assets
    # High goodwill ratio = future writedown risk. M&A-heavy firms accumulate
    # goodwill that can be impaired if acquisitions underperform.
    # >40% is typically considered elevated.
    goodwill_ta = None
    if gw is not None and ta is not None and ta > 0:
        goodwill_ta = round(gw / ta, 4)

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

    # ── Beneish M-Score (forensic earnings manipulation screen) ────────────────
    # Computed on annual (FY) data per Beneish 1999 canonical formulation.
    # Suppressed for banks/insurance/REITs (different accounting structure).
    def _beneish_pair(key):
        """Return (value_t, value_p) from annual series, or (None, None)."""
        s = raw_annual.get(key)
        if s is None or len(s) < 2:
            return (None, None)
        return (float(s.iloc[-1]), float(s.iloc[-2]))

    beneish = None
    if sector_bucket == 'industrial' and raw_annual:
        # Pull pairs from annual series
        rev_t, rev_p = _beneish_pair('revenue')
        recv_t, recv_p = _beneish_pair('receivables')
        gp_t, gp_p = _beneish_pair('gross_profit')
        # Batch 7h.22: same accounting-identity sanity check as the TTM path.
        # If extracted gross_profit is implausibly small relative to operating_income
        # for the same period (OI > GP violates definitional ordering), treat as
        # missing so the revenue - cost_of_revenue fallback can supply correct values.
        # This is the root-cause fix for META's GMI bug and AMZN's gross-profit display.
        oi_t_check, oi_p_check = _beneish_pair('operating_income')
        if gp_t is not None and oi_t_check is not None and oi_t_check > 0 and gp_t < 0.9 * oi_t_check:
            gp_t = None
        if gp_p is not None and oi_p_check is not None and oi_p_check > 0 and gp_p < 0.9 * oi_p_check:
            gp_p = None
        # Fallback: derive gross profit from revenue - cost_of_revenue if not direct
        if gp_t is None or gp_p is None:
            cor_t, cor_p = _beneish_pair('cost_of_revenue')
            if gp_t is None and rev_t is not None and cor_t is not None:
                gp_t = rev_t - cor_t
            if gp_p is None and rev_p is not None and cor_p is not None:
                gp_p = rev_p - cor_p
        ca_t, ca_p = _beneish_pair('current_assets')
        ppe_t, ppe_p = _beneish_pair('ppe_net')
        ta_t, ta_p = _beneish_pair('total_assets')
        dep_t, dep_p = _beneish_pair('da')  # depreciation+amortization expense
        gross_ppe_t, gross_ppe_p = _beneish_pair('gross_ppe')
        # Fallback: derive gross PPE from net PPE + accumulated depreciation
        if gross_ppe_t is None or gross_ppe_p is None:
            ad_t, ad_p = _beneish_pair('accumulated_depreciation')
            if ppe_t is not None and ad_t is not None:
                gross_ppe_t = ppe_t + ad_t
            if ppe_p is not None and ad_p is not None:
                gross_ppe_p = ppe_p + ad_p
        sga_t, sga_p = _beneish_pair('sga')
        ni_t, _ = _beneish_pair('net_income')
        cfo_t, _ = _beneish_pair('cfo')
        ltd_t, ltd_p = _beneish_pair('long_term_debt')
        cl_t, cl_p = _beneish_pair('current_liabilities')

        beneish_annual = compute_beneish_m_score(
            rev_t, rev_p, recv_t, recv_p, gp_t, gp_p,
            ca_t, ca_p, ppe_t, ppe_p, ta_t, ta_p,
            dep_t, dep_p, gross_ppe_t, gross_ppe_p,
            sga_t, sga_p, ni_t, cfo_t,
            ltd_t, ltd_p, cl_t, cl_p,
        )
        beneish = {'annual': beneish_annual, 'sector_eligible': True}
    else:
        # Banks, insurance, REITs: model not applicable
        beneish = {
            'annual': None,
            'sector_eligible': False,
            'note': f'Beneish M-Score not applicable to {sector_bucket} sector (different accounting structure)',
        }

    # ── Batch 7h.9: Sector-specific metrics ────────────────────────────────────
    sector_metrics = {}

    if sector_bucket == 'bank':
        # Net Interest Margin: NII / average earning assets (loans + investments).
        # Approximation: use loans_receivable as proxy when investments series unavailable.
        avg_earning_assets = loans_recv  # simple proxy
        nim = safe_div(net_int_income_ttm, avg_earning_assets) if avg_earning_assets else None
        # Efficiency ratio: noninterest expense / (NII + noninterest income). <60% = efficient.
        efficiency_ratio = None
        if noninterest_expense_ttm and (net_int_income_ttm or noninterest_income_ttm):
            denom = (net_int_income_ttm or 0) + (noninterest_income_ttm or 0)
            efficiency_ratio = safe_div(noninterest_expense_ttm, denom)
        # Tier 1 capital ratio: Tier1 capital / Risk-Weighted Assets. >10.5% = well capitalized.
        tier1_ratio = safe_div(tier1_cap, rwa)
        # Loans / Deposits: liquidity proxy. <80% conservative, >100% reaching.
        loan_deposit_ratio = safe_div(loans_recv, deposits_total)
        # NPL coverage: allowance / loans. ~1-2% = healthy reserve. <1% = under-reserved.
        nplc_ratio = safe_div(allowance_ll, loans_recv)
        # Pre-Provision Net Revenue: NII + non-int income - non-int expense
        ppnr = None
        if net_int_income_ttm is not None or noninterest_income_ttm is not None:
            ppnr = round(
                (net_int_income_ttm or 0) +
                (noninterest_income_ttm or 0) -
                (noninterest_expense_ttm or 0), 2
            )
        sector_metrics = {
            'sector': 'bank',
            'sector_label': 'Bank',
            'net_interest_income_ttm':  net_int_income_ttm,
            'noninterest_income_ttm':   noninterest_income_ttm,
            'noninterest_expense_ttm':  noninterest_expense_ttm,
            'provision_loan_losses_ttm': provision_ll_ttm,
            'pre_provision_net_revenue_ttm': ppnr,
            'loans_receivable':         loans_recv,
            'deposits_total':           deposits_total,
            'tier1_capital':            tier1_cap,
            'risk_weighted_assets':     rwa,
            'allowance_loan_losses':    allowance_ll,
            'net_interest_margin':      nim,             # NII / earning assets
            'efficiency_ratio':         efficiency_ratio, # cost-to-income
            'tier1_ratio':              tier1_ratio,      # capital adequacy
            'loan_to_deposit_ratio':    loan_deposit_ratio,
            'allowance_to_loans':       nplc_ratio,
        }

    elif sector_bucket == 'insurance':
        # Loss Ratio: losses incurred / premiums earned. Lower = better. Industry avg ~60-70%.
        loss_ratio = safe_div(losses_ttm, premiums_ttm)
        # Expense Ratio: underwriting expenses / premiums earned. Industry avg ~25-30%.
        expense_ratio = safe_div(underwrite_exp_ttm, premiums_ttm)
        # Combined Ratio: loss_ratio + expense_ratio. <100% = profitable underwriting.
        combined_ratio = None
        if loss_ratio is not None and expense_ratio is not None:
            combined_ratio = round(loss_ratio + expense_ratio, 4)
        # Investment yield: investment income / investments held
        inv_yield = safe_div(inv_income_ttm, investments_held)
        # Underwriting profit
        underwriting_profit = None
        if premiums_ttm is not None:
            underwriting_profit = round(
                premiums_ttm - (losses_ttm or 0) - (underwrite_exp_ttm or 0), 2
            )
        sector_metrics = {
            'sector': 'insurance',
            'sector_label': 'Insurance',
            'premiums_earned_ttm':  premiums_ttm,
            'losses_incurred_ttm':  losses_ttm,
            'underwriting_expenses_ttm': underwrite_exp_ttm,
            'underwriting_profit_ttm':   underwriting_profit,
            'investment_income_ttm':     inv_income_ttm,
            'investments_held':          investments_held,
            'reserves_total':            reserves_total,
            'loss_ratio':                loss_ratio,
            'expense_ratio':             expense_ratio,
            'combined_ratio':            combined_ratio,  # <1.00 = profitable
            'investment_yield':          inv_yield,
        }

    elif sector_bucket == 'reit':
        # Real Estate revenue (top-line)
        re_revenue_ttm = real_estate_revenue_ttm or rental_income_ttm or rev_ttm
        # NOI: ~ operating income for REITs (rental income - operating expenses)
        noi_ttm = oi_ttm
        # FFO (Funds From Operations) — non-GAAP industry standard
        ffo = ffo_ttm
        # FFO per share approximation (need shares outstanding)
        shares_out = last(raw.get('shares_outstanding'))
        ffo_per_share = safe_div(ffo, shares_out) if (ffo and shares_out) else None
        # Debt to gross real estate: leverage proxy
        debt_to_re = safe_div(mortgages_payable, real_estate_at_cost)
        # NOI yield: NOI / real estate at cost (income-on-cost)
        noi_yield = safe_div(noi_ttm, real_estate_at_cost)
        sector_metrics = {
            'sector': 'reit',
            'sector_label': 'REIT',
            'real_estate_revenue_ttm':  re_revenue_ttm,
            'rental_income_ttm':        rental_income_ttm,
            'noi_ttm':                  noi_ttm,
            'ffo_ttm':                  ffo,
            'ffo_per_share':            ffo_per_share,
            'real_estate_at_cost':      real_estate_at_cost,
            'real_estate_acc_dep':      real_estate_acc_dep,
            'real_estate_net':          real_estate_net,
            'mortgages_payable':        mortgages_payable,
            'debt_to_real_estate':      debt_to_re,
            'noi_yield':                noi_yield,
        }

    elif sector_bucket == 'other':
        # Asset managers, broker-dealers, holding companies. Industrial template
        # partially applies but several metrics (Altman Z, gross margin) are unreliable.
        sector_metrics = {
            'sector': 'other',
            'sector_label': 'Other Financial',
            'note': 'Sector uses non-standard XBRL templates. Some industrial metrics may be unreliable; rely primarily on the variance EWS and cash flow figures.',
        }

    elif sector_bucket == 'biotech':
        # Pharma/biotech (SIC 2834-2836, 8731). Pre-revenue or clinical-stage firms
        # have operating losses and negative gross margins by design — R&D burn before
        # product approval. Altman Z structurally flags them as distressed even when
        # cash-rich. Beneish is suppressed (handled by the industrial-only gate above).
        # Rely on cash runway, variance EWS, and pipeline-stage context (not in scope here).
        sector_metrics = {
            'sector': 'biotech',
            'sector_label': 'Pharma / Biotech',
            'note': 'Pharma and biotech firms with operating losses and pre-revenue stages do not fit standard distress models (Altman Z, Beneish). Rely on the variance EWS, cash position, runway, and clinical-stage context.',
        }

    else:  # industrial — default; no extra metrics, but tag for UI
        sector_metrics = {
            'sector': 'industrial',
            'sector_label': 'Industrial',
        }


    # 5. Structural EWS — variance-based score (paper-validated AUC = 0.86-0.96)
    #    Computed from daily price returns, NOT quarterly fundamentals.
    #    Reference: Malone 2026 (Filter Collapse paper P07; IRFA submission).
    # Batch 7h.16: window_days now derived from the user-selected quarter
    # window via quarters_to_trading_days(), so the slider actually controls
    # how much price history feeds the variance computation. Previously
    # hardcoded to 252 days, which made the slider non-functional for ISC.
    variance_window_days = quarters_to_trading_days(window)
    variance_score = None
    if mkt.get('full_price_series') is not None:
        variance_score = compute_variance_score(
            mkt['full_price_series'],
            window_days=variance_window_days,
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
        traj = compute_series_trajectory(s, window=window)
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

    # 6b. Batch 7h.17 (Tesla fix): compute EPS and Revenue Growth fallbacks from
    # EDGAR data when yfinance returns None. Tesla and similar tickers commonly
    # have yfinance gaps for these fields despite the underlying data being present
    # in 10-Q/10-K filings.
    #
    # EPS fallback: ni_ttm (in $M) × 1e6 / shares_outstanding
    # Source priority: yfinance trailingEps → EDGAR-computed
    eps_val = mkt.get('eps_trailing')
    if eps_val is None and ni_ttm is not None:
        shares_out = last(raw.get('shares_outstanding')) or mkt.get('shares_outstanding')
        if shares_out and shares_out > 0:
            eps_val = round((ni_ttm * 1e6) / shares_out, 2)

    # Revenue Growth YoY fallback: (TTM revenue / TTM revenue 4 quarters ago) - 1
    # Source priority: yfinance revenueGrowth → EDGAR-computed
    rev_growth_val = mkt.get('revenue_growth')
    if rev_growth_val is None:
        rev_series = raw.get('revenue')
        if rev_series is not None:
            v = rev_series.dropna() if hasattr(rev_series, 'dropna') else None
            if v is not None and len(v) >= 8:
                ttm_now   = float(v.iloc[-4:].sum())
                ttm_prior = float(v.iloc[-8:-4].sum())
                if ttm_prior > 0 and not (np.isnan(ttm_now) or np.isnan(ttm_prior)):
                    rev_growth_val = round((ttm_now - ttm_prior) / ttm_prior, 4)

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
            'stable':     f'For {ticker}, the variance EWS regime is currently classified as STABLE in our framework. Mean rolling variance = {score:.4f} (annualized vol ~{vol_pct}%); trend {trend_label}. The value falls within the framework\'s baseline calibration range (variance ~0.04-0.10) for non-distressed historical cases.',
            'elevated':   f'For {ticker}, the variance EWS regime is currently classified as ELEVATED in our framework. Mean rolling variance = {score:.4f} (annualized vol ~{vol_pct}%); trend {trend_label}. The value falls above the framework\'s typical stable-firm calibration range. Sector-peer comparison and trend monitoring are common next analytical steps.',
            'rising':     f'For {ticker}, the variance EWS regime is currently classified as RISING in our framework. Mean variance = {score:.4f} (annualized vol ~{vol_pct}%); trend rho = {trend:.2f}; latest/baseline ratio = {ratio:.2f}x. In the framework\'s historical calibration set, similar patterns appeared in cases including Lehman, AIG, and SVB; the framework does not predict outcomes for any specific company.',
            'distressed': f'For {ticker}, the variance EWS regime is currently classified as DISTRESSED in our framework — the highest classification in our framework\'s scale. Mean variance = {score:.4f} (annualized vol ~{vol_pct}%); trend rho = {trend:.2f}; latest/baseline ratio = {ratio:.2f}x. This combination of magnitude and trajectory falls within the range observed in the framework\'s pre-collapse calibration cases; the framework does not predict outcomes for any specific company.',
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
    # Batch 7h.20: sector/industry display fallback chain.
    # Primary: yfinance info.sector/industry (when available).
    # Fallback: EDGAR sic_description (always available for US filers).
    # Without this fallback, the UI shows a blank sector line for any ticker
    # where yfinance is blocked/rate-limited on Render, which is most of them.
    display_sector = mkt.get('sector') or sic_description or ''
    display_industry = mkt.get('industry') or sic_description or ''

    result = {
        'ticker':       ticker,
        'company_name': mkt.get('company_name', ticker),
        'analysis_mode': 'edgar',  # Batch 7h.14: explicit mode flag for frontend
        'window':       window,    # Batch 7h.15: quarters used for trajectory analysis
        'sector':       display_sector,    # yfinance preferred; EDGAR sicDescription fallback
        'industry':     display_industry,
        # Batch 7h.9: structural sector classification from SIC code
        'sector_bucket':    sector_bucket,        # one of: industrial, bank, insurance, reit, other
        'sic_code':         sic_code,
        'sic_description':  sic_description,
        'sector_metrics':   sector_metrics,
        'description':  mkt.get('description', ''),
        'analysis_date':str(pd.Timestamp.now().date()),
        'data_sources': {
            'edgar': facts is not None,
            'yfinance': bool(mkt.get('price')) and HAS_YF,  # legacy field for frontend compat
            'prices': mkt.get('price_source', 'failed'),    # 'tiingo' | 'stooq' | 'failed'
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
            'gross_margin':     {'val': round(gross_margin*100, 1) if gross_margin is not None else None, 'label': 'Gross Margin %', 'simple': 'What fraction of each sale is kept after direct costs'},
            'operating_income': {'val': oi_ttm,     'label': 'Operating Income (TTM $M)',  'simple': 'Profit from running the business — before debt and taxes'},
            'operating_margin': {'val': round(op_margin*100, 1) if op_margin is not None else None, 'label': 'Operating Margin %', 'simple': 'Operating profit as % of sales'},
            'ebitda':           {'val': ebitda_ttm, 'label': 'EBITDA (TTM $M)',             'simple': 'Cash-like profit before accounting adjustments'},
            'net_income':       {'val': ni_ttm,     'label': 'Net Income (TTM $M)',         'simple': 'What the company kept after every cost — the bottom line'},
            'net_margin':       {'val': round(net_margin*100, 1) if net_margin is not None else None, 'label': 'Net Margin %', 'simple': 'Final profit as % of sales'},
            'interest_expense': {'val': int_ttm,    'label': 'Interest Expense (TTM $M)',   'simple': 'What the company pays on its debt each year'},
            'da':               {'val': da_ttm,     'label': 'D&A (TTM $M)',               'simple': 'Paper cost of wearing out assets — not actual cash going out'},
            'eps':              {'val': eps_val, 'label': 'EPS (Trailing)', 'simple': 'Profit per share — what each share earned'},
            'revenue_growth':   {'val': round(rev_growth_val*100, 1) if rev_growth_val is not None else None, 'label': 'Revenue Growth YoY %', 'simple': 'How fast is the top line growing?'},
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
            'fcf_margin':    {'val': round(fcf_margin*100, 1) if fcf_margin is not None else None, 'label': 'FCF Margin %', 'simple': 'Free cash flow as % of revenue'},
            'cash_conversion':{'val': cash_conv,'label': 'Cash Conversion (CFO/NI)',     'simple': 'Is reported profit backed by actual cash? Above 1.0 is good'},
        },

        # Batch 7h.10: quarterly history arrays for time-series rendering.
        # Each entry is a list of {date, val} dicts, oldest-to-newest, up to last 12 quarters.
        'quarterly_history': _build_quarterly_history(raw),


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
            'roic':            {'val': round((roic or 0)*100,2) if roic is not None else None, 'label': 'ROIC %', 'simple': 'Return on invested capital. NOPAT \u00f7 (debt + equity). Less distorted by buybacks than ROE. Above 15% is excellent.', 'effective_tax_rate': round(effective_tax_rate * 100, 1), 'effective_tax_rate_source': effective_tax_rate_source},
            'asset_turnover':  {'val': asset_turn,  'label': 'Asset Turnover',     'simple': 'How much revenue does each dollar of assets generate?'},
            'gross_margin_pct':{'val': round((gross_margin or 0)*100,1) if gross_margin else None, 'label': 'Gross Margin %', 'simple': 'Fraction of each sale kept after direct costs'},
            'op_margin_pct':   {'val': round((op_margin or 0)*100,1) if op_margin else None,   'label': 'Operating Margin %', 'simple': 'Operating profit as fraction of sales'},
            'ebitda_margin_pct': {'val': round((ebitda_margin or 0)*100,1) if ebitda_margin is not None else None, 'label': 'EBITDA Margin %', 'simple': 'EBITDA as fraction of revenue. Captures operating profitability before depreciation and amortization.'},
            'fcf_yield_pct':   {'val': round((fcf_yield or 0)*100,2) if fcf_yield is not None else None, 'label': 'FCF Yield %', 'simple': 'Free cash flow divided by market cap. The cash a shareholder effectively earns per dollar invested. Inverse of P/FCF.'},
            'goodwill_ta_pct': {'val': round((goodwill_ta or 0)*100,1) if goodwill_ta is not None else None, 'label': 'Goodwill / Total Assets %', 'simple': 'Share of assets that came from acquisitions. Above 40% means future writedown risk if acquisitions underperform.'},
            'dso':             {'val': dso, 'label': 'Days Sales Outstanding', 'simple': 'How many days of revenue sit in receivables. Rising DSO can signal aggressive revenue recognition or weakening customer credit.'},
            'dio':             {'val': dio, 'label': 'Days Inventory Outstanding', 'simple': 'How many days of COGS sit in inventory. Rising DIO can signal demand weakening or inventory buildup before a writedown.'},
            'dpo':             {'val': dpo, 'label': 'Days Payable Outstanding', 'simple': 'How many days to pay suppliers. Rising DPO can signal cash stress (stretching payments) or improved leverage.'},
            'cash_conversion_cycle': {'val': ccc, 'label': 'Cash Conversion Cycle', 'simple': 'DSO + DIO \u2212 DPO. Days cash is tied up in operations. Negative is excellent (collect before paying). Watch the trend.'},
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
            'eps':           eps_val,
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

        'beneish': beneish,
    }

    return JSONResponse(content=clean_json(result))

@app.get("/health")
async def health():
    return {"status": "ok", "yfinance": HAS_YF}
@app.get("/diagnose")
async def diagnose(ticker: str = "AAPL"):
    import traceback
    import os
    from price_data import _get_tiingo_token
    TIINGO_TOKEN = _get_tiingo_token()
    result = {
        "ticker": ticker,
        "env_check": {
            "TIINGO_TOKEN_set": bool(TIINGO_TOKEN),
            "TIINGO_TOKEN_length": len(TIINGO_TOKEN) if TIINGO_TOKEN else 0,
            "TIINGO_TOKEN_first_4": TIINGO_TOKEN[:4] if TIINGO_TOKEN else "",
            "TIINGO_TOKEN_last_4": TIINGO_TOKEN[-4:] if TIINGO_TOKEN else "",
            "all_env_keys_with_tiingo_or_token": sorted([k for k in os.environ.keys() if 'tiingo' in k.lower() or 'token' in k.lower()]),
            "raw_env_lookup_TIINGO_TOKEN": os.environ.get('TIINGO_TOKEN', '<not_present>'),
            "all_env_keys_count": len(os.environ.keys()),
        },
        "tiingo_test": {},
        "stooq_test": {},
        "yfinance_test": {},
    }
    if TIINGO_TOKEN:
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{ticker.lower()}/prices"
            r = requests.get(url, params={
                "startDate": "2025-01-01",
                "endDate":   "2025-12-31",
                "token":     TIINGO_TOKEN,
            }, timeout=10)
            result["tiingo_test"] = {
                "status_code": r.status_code,
                "response_size": len(r.text),
                "response_preview": r.text[:200],
                "headers_content_type": r.headers.get("content-type", ""),
            }
        except Exception as e:
            result["tiingo_test"] = {"exception": str(e), "traceback": traceback.format_exc()[-500:]}
    else:
        result["tiingo_test"] = {"skipped": "TIINGO_TOKEN not set"}
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
        r = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        result["stooq_test"] = {
            "status_code": r.status_code,
            "response_size": len(r.text),
            "response_preview": r.text[:200],
        }
    except Exception as e:
        result["stooq_test"] = {"exception": str(e), "traceback": traceback.format_exc()[-500:]}
    if HAS_YF:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d", auto_adjust=True)
            result["yfinance_test"] = {
                "n_rows": len(hist),
                "columns": list(hist.columns) if not hist.empty else [],
                "latest_close": float(hist["Close"].iloc[-1]) if not hist.empty else None,
            }
        except Exception as e:
            result["yfinance_test"] = {"exception": str(e)}
    else:
        result["yfinance_test"] = {"skipped": "yfinance not installed"}
    return result
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
