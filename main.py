# Backend window-param wiring fix
**File:** `main.py`
**Purpose:** Make the quarter slider actually control ISC variance + fetch enough price history to support it.

---

## Change 1 — Add helper function (NEW)

**Add this near the top of `main.py`, anywhere after imports and before `compute_series_trajectory`.** A good spot is right above `def compute_series_trajectory(...)`.

```python
# ── Window translation helpers ────────────────────────────────────────────────
def quarters_to_trading_days(quarters: int) -> int:
    """Translate the frontend's 'Rolling quarters' slider value into a number of
    trading days for price-based variance computation.

    Roughly 63 trading days per fiscal quarter (252 trading days / 4 quarters).
    We add a small floor so very short windows still have enough history for
    a meaningful 90-day rolling variance computation.

    Args:
        quarters: Slider value (4–20 supported in current UI)
    Returns:
        Trading days to use as `window_days` in compute_variance_score().
        Floor of 180 trading days enforced so rolling-90 variance has at least
        ~3 months of headroom for the trend regression.
    """
    try:
        q = int(quarters) if quarters is not None else 12
    except (TypeError, ValueError):
        q = 12
    q = max(4, min(20, q))
    return max(180, q * 63)
```

---

## Change 2 — Bump price fetch from 2 years to 5 years

**Find this block** in `get_market_data()`:

```python
    # 1. Daily prices via Tiingo→Stooq fallback chain
    full_prices, price_source = fetch_daily_prices(ticker, days=730)
```

**Replace with:**

```python
    # 1. Daily prices via Tiingo→Stooq fallback chain
    # Batch 7h.16: bumped from 730 → 1825 days so the quarter-window slider
    # (max 20Q ≈ 1260 trading days ≈ 1825 calendar days) has the price history
    # it needs for long-window variance computation.
    full_prices, price_source = fetch_daily_prices(ticker, days=1825)
```

---

## Change 3 — Fix variance call in price-only mode

**Find this block** in `_build_price_only_response()`:

```python
    variance_score_pr = None
    full_series = mkt.get('full_price_series')
    if full_series is not None and len(full_series) >= 90:
        variance_score_pr = compute_variance_score(
            full_series,
            window_days=252,
            rolling_window=90,
        )
```

**Replace with:**

```python
    # Batch 7h.16: thread the user-selected window through to variance computation.
    # The frontend slider sends `window` in quarters; translate to trading days
    # so compute_variance_score respects the user's selection. Previously the
    # slider was UI theater — the variance regime never changed regardless of
    # slider position because window_days was hardcoded to 252.
    variance_window_days = quarters_to_trading_days(window)
    variance_score_pr = None
    full_series = mkt.get('full_price_series')
    if full_series is not None and len(full_series) >= 90:
        variance_score_pr = compute_variance_score(
            full_series,
            window_days=variance_window_days,
            rolling_window=90,
        )
```

---

## Change 4 — Fix variance call in main EDGAR-mode analyze route

**Find this block** in `analyze()` (after the section comment about Structural EWS):

```python
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
```

**Replace with:**

```python
    # 5. Structural EWS — variance-based score (paper-validated AUC = 0.86-0.96)
    #    Computed from daily price returns, NOT quarterly fundamentals.
    #    Reference: Malone 2026 (Filter Collapse paper P07; IRFA submission).
    #
    # Batch 7h.16: window_days now derived from the user-selected quarter window
    # via quarters_to_trading_days(), so the slider actually controls how much
    # price history is used for variance. Previously hardcoded to 252 days,
    # which made the slider non-functional for ISC variance.
    variance_window_days = quarters_to_trading_days(window)
    variance_score = None
    if mkt.get('full_price_series') is not None:
        variance_score = compute_variance_score(
            mkt['full_price_series'],
            window_days=variance_window_days,
            rolling_window=90,
        )
```

---

## Change 5 — Update default window in route signature

**Find this line** at the top of the route handler:

```python
@app.get("/analyze/{ticker}")
async def analyze(ticker: str, window: int = 6, mode: str = "edgar"):
```

**Replace with:**

```python
@app.get("/analyze/{ticker}")
async def analyze(ticker: str, window: int = 12, mode: str = "edgar"):
```

---

## Summary
| Change | Where | What it does |
|---|---|---|
| 1 | New function | Translates slider quarters → variance trading days |
| 2 | `get_market_data` | Fetches 5yr of prices instead of 2yr |
| 3 | `_build_price_only_response` | Honors window param in price-only mode |
| 4 | `analyze()` route | Honors window param in EDGAR mode (the real fix) |
| 5 | `analyze()` signature | Default window 6Q → 12Q |

## Validation after deploy
1. Open Network tab on AAPL
2. Move slider to 4Q → request `?window=4` → response has different ISC variance
3. Move slider to 12Q → request `?window=12` → different ISC variance
4. Move slider to 20Q → request `?window=20` → different ISC variance
5. Stability score should populate when window ≥ ~9Q (depends on stability function's internal threshold)
