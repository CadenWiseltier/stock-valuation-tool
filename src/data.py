"""
Data retrieval layer.

This module is the ONLY place in the application that talks to an external
financial-data provider (currently `yfinance`, which sources data from Yahoo
Finance). Every other module receives clean pandas DataFrames / plain Python
objects and never imports `yfinance` directly.

Why isolate this: Yahoo Finance is a free, unofficial data source with no
uptime guarantee, occasional missing fields, and rate limits. If a company
project outgrows it (e.g. moves to a paid provider such as Financial
Modeling Prep, Alpha Vantage, or a Bloomberg/FactSet feed), only this file
needs to change -- the rest of the app depends on the `StockDataBundle`
schema defined below, not on yfinance's data shapes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


class TickerNotFoundError(Exception):
    """Raised when a ticker symbol does not resolve to a tradable company."""


class DataProviderError(Exception):
    """Raised when the data provider could not be reached, or refused the
    request (rate limiting, an outage, no network).

    Kept DISTINCT from `TickerNotFoundError` because the two need opposite
    responses from the user, and conflating them produces a confidently
    wrong message. Yahoo Finance is a free, unofficial, rate-limited source:
    when it throttles a caller, every request fails at once, which used to
    surface as "'AAPL' could not be resolved to a valid, tradable stock --
    check the ticker symbol." That tells a user to fix a ticker that was
    never wrong, and on a public site where many visitors share one server
    IP, throttling is the single most likely failure mode.
    """


# ---------------------------------------------------------------------------
# Canonical statement row labels.
#
# yfinance's statement schema has shifted slightly across versions/companies,
# so each line item is looked up (via `row()`, below) by trying several
# known spellings in order. Centralized here so financial_analysis.py and
# dcf.py -- which both need to read the same raw line items -- share one
# source of truth instead of duplicating or cross-importing private lists.
# ---------------------------------------------------------------------------
REVENUE = ["Total Revenue", "Operating Revenue"]
GROSS_PROFIT = ["Gross Profit"]
OPERATING_INCOME = ["Operating Income"]
NET_INCOME = ["Net Income Common Stockholders", "Net Income", "Net Income Continuous Operations"]
DILUTED_EPS = ["Diluted EPS"]
BASIC_EPS = ["Basic EPS"]
DILUTED_SHARES = ["Diluted Average Shares", "Basic Average Shares"]
RESEARCH_AND_DEVELOPMENT = ["Research And Development"]
EBIT = ["EBIT"]
EBITDA = ["EBITDA"]
PRETAX_INCOME = ["Pretax Income"]
TAX_PROVISION = ["Tax Provision"]
INTEREST_EXPENSE = ["Interest Expense", "Interest Expense Non Operating"]

OPERATING_CASH_FLOW = ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
CAPEX = ["Capital Expenditure", "Purchase Of Property Plant And Equipment"]
D_AND_A_CF = ["Depreciation Amortization Depletion", "Depreciation And Amortization"]

CASH = ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]
TOTAL_DEBT = ["Total Debt"]
CURRENT_ASSETS = ["Current Assets"]
CURRENT_LIABILITIES = ["Current Liabilities"]
TOTAL_ASSETS = ["Total Assets"]
TOTAL_EQUITY = ["Stockholders Equity", "Total Equity Gross Minority Interest"]
LONG_TERM_DEBT = ["Long Term Debt"]
CURRENT_DEBT = ["Current Debt", "Current Debt And Capital Lease Obligation"]


# ---------------------------------------------------------------------------
# Sector -> peer-group fallback map.
#
# Automatic peer discovery (via yfinance sector/industry metadata) is
# attempted first. When that is unavailable or too sparse, the application
# falls back to this curated list of large, liquid, well-known companies in
# the same GICS-style sector bucket as reported by the data provider. This
# keeps comparable-company analysis meaningful (e.g. we never compare a
# software company against an oil refiner) without requiring a paid
# screener API.
# ---------------------------------------------------------------------------
SECTOR_PEER_MAP: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "ORCL", "CRM", "ADBE", "AVGO"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "CMCSA"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MDLZ"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BRK-B"],
    "Industrials": ["HON", "UPS", "CAT", "BA", "GE", "LMT", "RTX", "MMM"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "PSA", "O"],
    "Basic Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM"],
}


@dataclass
class StockDataBundle:
    """Container for all raw data pulled for a single ticker.

    Every field is Optional / can be an empty DataFrame -- downstream
    modules must tolerate missing data and surface "N/A" rather than
    crashing or fabricating numbers (see src/financial_analysis.py).
    """

    ticker: str
    info: dict = field(default_factory=dict)
    income_stmt: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_sheet: pd.DataFrame = field(default_factory=pd.DataFrame)
    cash_flow: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_as_of: Optional[pd.Timestamp] = None


def _safe_get(d: dict, *keys, default=None):
    """Return the first present, non-None value among `keys` in dict `d`."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


#: Substrings that identify a throttling or connectivity failure rather than
#: a genuine "no such company" answer. Matched case-insensitively against the
#: exception text, which is the only signal yfinance surfaces -- it raises
#: plain exceptions rather than typed HTTP errors.
_RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "rate-limit", "throttl")
_CONNECTIVITY_MARKERS = ("could not resolve host", "connection", "timed out", "timeout",
                          "temporary failure", "network is unreachable", "ssl", "max retries")


def _describe_provider_failure(errors: list[Exception]) -> str:
    """Turn provider exceptions into one sentence a non-technical visitor can
    act on. Rate limiting and loss of connectivity need different advice, and
    both need different advice from a mistyped ticker."""
    blob = " ".join(str(e) for e in errors).lower()
    if any(m in blob for m in _RATE_LIMIT_MARKERS):
        return ("The market-data provider is currently rate-limiting requests. This is a limit on "
                "the free data source, not a problem with the ticker you entered. Please wait a "
                "minute and try again.")
    if any(m in blob for m in _CONNECTIVITY_MARKERS):
        return ("Could not reach the market-data provider. This is usually a temporary network or "
                "provider outage rather than a problem with the ticker you entered. Please try "
                "again shortly.")
    return ("The market-data provider did not return any data for this request. This is usually "
            "temporary. Please try again shortly.")


def _normalize_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a yfinance financial-statement DataFrame oldest -> newest columns.

    yfinance returns statements with the most recent fiscal period as the
    first column. Chronological (oldest-first) ordering is easier to reason
    about for growth-rate and trend calculations, so we standardize on it
    here once rather than re-sorting in every downstream function.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.loc[:, sorted(out.columns)]
    return out


def fetch_stock_data(ticker: str, years: int = 5) -> StockDataBundle:
    """Fetch all raw data needed for analysis of a single ticker.

    Parameters
    ----------
    ticker: stock ticker symbol, e.g. "AMZN"
    years: number of years of price history to retrieve (statements are
        limited by whatever the data provider makes available, typically
        4 annual periods for the free tier).

    Raises
    ------
    TickerNotFoundError if the ticker does not resolve to a real, tradable
    company (e.g. typo, delisted, or invalid symbol).
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise TickerNotFoundError("No ticker was provided.")

    t = yf.Ticker(ticker)

    # Every provider failure is recorded so that "we could not reach the
    # provider" can be told apart from "the provider answered, and this
    # ticker does not exist" -- see DataProviderError.
    provider_errors: list[Exception] = []

    try:
        info = t.get_info()
    except Exception as exc:
        provider_errors.append(exc)
        info = {}

    # A valid company should have at least a short name or a regular market
    # price. If neither is present, yfinance most likely returned an empty
    # shell for an invalid/delisted ticker.
    has_name = bool(_safe_get(info, "shortName", "longName"))
    has_price = _safe_get(info, "currentPrice", "regularMarketPrice") is not None

    # auto_adjust=True so "Close" is split/dividend-adjusted. This matters
    # beyond display: historical P/E (src/financial_analysis.py:
    # historical_pe_series) divides a historical price by that period's
    # reported EPS, and data providers restate historical EPS on a
    # post-split basis -- an un-adjusted historical Close would be wrong by
    # the split factor for any company that split its stock during the
    # lookback window (e.g. AMZN's 20:1 split in 2022).
    try:
        history = t.history(period=f"{max(years, 1)}y", auto_adjust=True)
    except Exception as exc:
        provider_errors.append(exc)
        history = pd.DataFrame()

    if not has_name and not has_price and history.empty:
        # Nothing came back. Which of the two reasons it was decides what the
        # user should be told, so guessing here is not acceptable.
        if provider_errors:
            raise DataProviderError(_describe_provider_failure(provider_errors)) from provider_errors[0]
        raise TickerNotFoundError(
            f"'{ticker}' could not be resolved to a valid, tradable stock. "
            "Check the ticker symbol and try again."
        )

    def _try(getter):
        try:
            return _normalize_statement(getter())
        except Exception:
            return pd.DataFrame()

    income_stmt = _try(lambda: t.income_stmt)
    balance_sheet = _try(lambda: t.balance_sheet)
    cash_flow = _try(lambda: t.cashflow)

    data_as_of = history.index.max() if not history.empty else pd.Timestamp.utcnow()

    return StockDataBundle(
        ticker=ticker,
        info=info or {},
        income_stmt=income_stmt,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        price_history=history,
        data_as_of=data_as_of,
    )



# ---------------------------------------------------------------------------
# Live price refresh.
#
# `fetch_stock_data` is deliberately cached for an hour by the UI: financial
# statements change quarterly at most, and refetching them on every
# interaction would burn through a free provider's rate limit immediately.
# The PRICE, however, is the one input that must not be an hour old --
# roughly 65% of the Fundamental Investment Score is price-dependent
# (src/scoring.py), so a stale quote moves the RATING, not merely a displayed
# number. Measured on AMZN, the score shifts about 2.35 points per 1% of
# price movement.
#
# These helpers therefore fetch ONLY the last trade price, cheaply enough to
# sit on a short cache, independently of the statement bundle.
#
# Two independent providers are queried so the answer can be cross-checked.
# Agreement between unrelated sources is the only evidence available here
# that a quote is real rather than a stale or malformed field, which is why
# the second source is deliberately not Yahoo: two figures from one provider
# agree even when that provider is wrong.
#
# Both endpoints return JSON intended for programmatic use. Screen-scraping a
# retail finance site (MarketWatch, Yahoo's HTML, CNBC) is deliberately NOT
# done -- those pages carry no usage grant, change layout without notice, and
# block datacenter IPs, which is exactly where a deployed Streamlit app runs
# from. Stooq's CSV quote endpoint was the first choice for the cross-check
# and had to be dropped: it now answers with a JavaScript bot-verification
# page instead of data, which is the same failure mode a scraper would hit.
# ---------------------------------------------------------------------------

#: A cross-source gap wider than this is reported rather than silently
#: averaged away. 0.5% is comfortably wider than ordinary bid/ask and
#: quote-timing noise, and narrow enough to catch a genuinely wrong figure.
LIVE_PRICE_DISAGREEMENT_TOLERANCE = 0.005

#: How old a quote may be before it is labelled stale. Generous, because
#: outside market hours the newest real quote IS the previous close.
LIVE_PRICE_STALE_AFTER = pd.Timedelta(hours=1)


@dataclass
class LivePrice:
    """A last-trade price together with the provenance needed to judge it.

    `price` is never silently blended between providers: it is whichever
    source answered first, and `cross_check_price` records what the other
    said. A caller that wants to refuse to act on disagreeing sources can
    read `sources_disagree`; one that just wants a number can ignore it.
    """

    price: float
    source: str
    as_of: Optional[pd.Timestamp] = None
    currency: str = "USD"
    cross_check_price: Optional[float] = None
    cross_check_source: Optional[str] = None

    @property
    def disagreement(self) -> Optional[float]:
        """Fractional gap between the two sources, or None if only one answered."""
        if self.cross_check_price is None or not self.price:
            return None
        return abs(self.cross_check_price - self.price) / self.price

    @property
    def sources_disagree(self) -> bool:
        d = self.disagreement
        return d is not None and d > LIVE_PRICE_DISAGREEMENT_TOLERANCE

    @property
    def age(self) -> Optional[pd.Timedelta]:
        if self.as_of is None:
            return None
        ts = pd.Timestamp(self.as_of)
        now = pd.Timestamp.now(tz=ts.tz) if ts.tz is not None else pd.Timestamp.utcnow()
        return now - ts

    @property
    def is_stale(self) -> bool:
        """True when the quote is older than LIVE_PRICE_STALE_AFTER.

        An unknown timestamp returns False, not True: "we cannot tell how old
        this is" is a different claim from "this is old", and reporting the
        latter would fabricate a fact about the data.
        """
        age = self.age
        return age is not None and age > LIVE_PRICE_STALE_AFTER


def _coerce_price(value) -> Optional[float]:
    """Return a strictly positive finite float, or None.

    Providers signal "no data" with 0, "N/D", "-", empty string and NaN in
    roughly equal measure. Every one of those must become None rather than a
    zero price, which downstream would be read as a real quote and produce
    an infinite upside.
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Nasdaq returns prices pre-formatted for display ("$258.51"), and
        # thousands separators appear once a share price runs to four digits.
        value = value.replace("$", "").replace(",", "").strip()
        if not value or value.upper() in ("N/A", "N/D", "-", "--"):
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f <= 0:
        return None
    return f


def live_price_from_yfinance(ticker: str) -> Optional[LivePrice]:
    """Last trade price from Yahoo Finance via yfinance's quote endpoint.

    Prefers `fast_info`, a far smaller request than the full `get_info()`
    payload `fetch_stock_data` makes -- that size difference is what makes
    this affordable to call on a short cache.
    """
    t = yf.Ticker(ticker)
    price = None
    as_of = None

    try:
        fast = t.fast_info
        raw = getattr(fast, "last_price", None)
        if raw is None and hasattr(fast, "get"):
            raw = fast.get("lastPrice")
        price = _coerce_price(raw)
    except Exception:
        price = None

    if price is None:
        try:
            info = t.get_info()
        except Exception:
            return None
        price = _coerce_price(_safe_get(info, "currentPrice", "regularMarketPrice"))
        raw_ts = info.get("regularMarketTime")
        if raw_ts is not None:
            try:
                as_of = pd.to_datetime(raw_ts, unit="s", utc=True)
            except (ValueError, TypeError, OverflowError):
                as_of = None

    if price is None:
        return None
    return LivePrice(price=price, source="Yahoo Finance", as_of=as_of)


def live_price_from_nasdaq(ticker: str) -> Optional[LivePrice]:
    """Last sale price from Nasdaq's public quote API.

    Used as the cross-check because it is genuinely independent of Yahoo --
    it comes from the listing exchange itself -- returns JSON, needs no API
    key, and covers NYSE-listed names as well as Nasdaq-listed ones.

    Its feed is delayed, and it says so via `isRealTime`; that flag is
    carried through rather than assumed, since a delayed cross-check still
    catches a wrong price even though it cannot confirm an intraday one.
    """
    import requests

    symbol = ticker.strip().upper()

    try:
        resp = requests.get(
            "https://api.nasdaq.com/api/quote/{0}/info".format(symbol),
            params={"assetclass": "stocks"},
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; stock-valuation-tool/1.0)"},
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    data = payload.get("data") or {}
    primary = data.get("primaryData") or {}

    price = _coerce_price(primary.get("lastSalePrice"))
    if price is None:
        return None

    as_of = None
    stamp = primary.get("lastTradeTimestamp")
    if stamp:
        # e.g. "Sep 3, 2026" or "Sep 3, 2026 4:00 PM ET". The trailing
        # exchange-timezone label is not parseable by pandas, so it is
        # stripped; an unparseable stamp yields None rather than a guess.
        cleaned = re.sub(r"\s*(ET|EST|EDT)\b.*$", "", str(stamp)).strip()
        cleaned = re.sub(r"^(LAST TRADE|AS OF)[:\s]*", "", cleaned, flags=re.I).strip()
        try:
            as_of = pd.to_datetime(cleaned)
        except (ValueError, TypeError):
            as_of = None

    return LivePrice(price=price, source="Nasdaq", as_of=as_of)


#: Queried in order. The first to answer supplies the price, the next to
#: answer becomes the cross-check. Append a callable to add a provider --
#: it takes a ticker and returns a LivePrice or None, and must not raise.
LIVE_PRICE_SOURCES = (live_price_from_yfinance, live_price_from_nasdaq)


def fetch_live_price(ticker: str, sources=None, cross_check: bool = True) -> Optional[LivePrice]:
    """Fetch the current trade price for `ticker`, cross-checked where possible.

    Returns None when no source answered. It never raises and never
    substitutes a guess, because the caller's correct response to "no live
    price available" is to keep using the bundle's cached price and say so
    on screen -- not to abort a report the user has already waited for.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None

    results = []
    for source in (LIVE_PRICE_SOURCES if sources is None else sources):
        try:
            got = source(ticker)
        except Exception:
            got = None
        if got is None:
            continue
        results.append(got)
        if not cross_check or len(results) == 2:
            break

    if not results:
        return None

    primary = results[0]
    if len(results) > 1:
        primary.cross_check_price = results[1].price
        primary.cross_check_source = results[1].source
    return primary


def apply_live_price(bundle: StockDataBundle, live: Optional[LivePrice]) -> StockDataBundle:
    """Overwrite the bundle's hour-old quote with a freshly fetched one.

    Mutates and returns `bundle`.

    Market cap and enterprise value are rescaled by the same move rather
    than left alone. They are price times share count, so leaving them
    stale while the price updates would make the report internally
    inconsistent: WACC's equity weight (src/dcf.py) and the Speculation
    Score's upside multiple (src/speculative.py) both read `marketCap`, and
    would otherwise be describing a different price from the one on screen.
    Net debt is held constant, which is precisely what moving only the
    equity leg of the valuation means.
    """
    if live is None or not live.price:
        return bundle

    info = bundle.info
    old_price = _coerce_price(_safe_get(info, "currentPrice", "regularMarketPrice"))

    info["currentPrice"] = live.price
    info["regularMarketPrice"] = live.price
    info["livePriceSource"] = live.source
    if live.as_of is not None:
        info["livePriceAsOf"] = live.as_of

    shares = _coerce_price(info.get("sharesOutstanding"))
    old_market_cap = _coerce_price(info.get("marketCap"))

    if shares is not None:
        new_market_cap = live.price * shares
    elif old_market_cap is not None and old_price is not None:
        new_market_cap = old_market_cap * (live.price / old_price)
    else:
        new_market_cap = None

    if new_market_cap is not None:
        old_ev = _coerce_price(info.get("enterpriseValue"))
        if old_ev is not None and old_market_cap is not None:
            # EV = market cap + net debt. Hold net debt, move the equity leg.
            info["enterpriseValue"] = old_ev + (new_market_cap - old_market_cap)
        info["marketCap"] = new_market_cap

    return bundle


# ---------------------------------------------------------------------------
# Business-summary-based sector override for peer selection.
#
# The data provider's own "sector"/"industry" classification is sometimes
# unreliable for newer or structurally unusual listings -- found in testing
# with IREN Limited, which Yahoo Finance classifies as "Financial Services
# / Capital Markets" (likely inherited from an early corporate structure)
# even though its own reported business summary describes it as an AI
# cloud-computing / data-center operator. Building a peer group from the
# REPORTED sector in a case like this compares the company against
# megabanks -- a confidently wrong comparison, not an honest "N/A".
#
# This is a narrow, keyword-based override for a small set of well-known,
# high-signal business descriptions where the reported sector is likely to
# mislead peer selection -- NOT a general text classifier. It only ever
# fires when a keyword is found AND the implied sector differs from the
# reported one, and it never changes what is *displayed* as the company's
# sector/industry elsewhere in the app -- only which peer basket is used.
# ---------------------------------------------------------------------------
BUSINESS_SUMMARY_SECTOR_OVERRIDES: list[tuple[list[str], str]] = [
    (["data center", "data centers", "cloud computing", "cloud services",
      "ai cloud", "artificial intelligence"], "Technology"),
    (["semiconductor"], "Technology"),
    (["bitcoin mining", "cryptocurrency mining", "digital asset mining", "crypto mining"], "Technology"),
]


def infer_peer_sector(info: dict) -> Optional[str]:
    """Return the sector to use for PEER SELECTION, which may differ from
    the data provider's reported `info["sector"]` -- see
    BUSINESS_SUMMARY_SECTOR_OVERRIDES above.
    """
    reported_sector = info.get("sector")
    summary = (info.get("longBusinessSummary") or "").lower()
    for keywords, override_sector in BUSINESS_SUMMARY_SECTOR_OVERRIDES:
        if override_sector != reported_sector and any(kw in summary for kw in keywords):
            return override_sector
    return reported_sector


def get_peer_tickers(bundle: StockDataBundle, max_peers: int = 6) -> list[str]:
    """Return a small list of comparable-company tickers for `bundle`.

    Strategy: use the company's sector (as classified by the data provider,
    subject to the override in `infer_peer_sector` above) to select from a
    curated peer map of large, liquid companies in the same sector,
    excluding the company itself. This is a deliberately simple and
    transparent substitute for a paid peer-screening API -- see
    docs/methodology.md for the rationale.
    """
    sector = infer_peer_sector(bundle.info)
    peers = SECTOR_PEER_MAP.get(sector, [])
    peers = [p for p in peers if p.upper() != bundle.ticker.upper()]
    return peers[:max_peers]


def fetch_peer_snapshot(ticker: str) -> dict:
    """Fetch just the lightweight valuation-multiple fields needed for one
    peer in the comparable-company table (not full statements, to keep peer
    lookups fast).
    """
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()
    except Exception:
        info = {}
    return {
        "ticker": ticker,
        "name": _safe_get(info, "shortName", "longName", default=ticker),
        "trailing_pe": _safe_get(info, "trailingPE"),
        "ev_to_ebitda": _safe_get(info, "enterpriseToEbitda"),
        "ev_to_revenue": _safe_get(info, "enterpriseToRevenue"),
        "price_to_sales": _safe_get(info, "priceToSalesTrailing12Months"),
        "market_cap": _safe_get(info, "marketCap"),
        "enterprise_value": _safe_get(info, "enterpriseValue"),
    }


def row(df: pd.DataFrame, *labels) -> Optional[pd.Series]:
    """Look up the first matching row (by exact label) in a statement
    DataFrame. yfinance statement row labels are fairly stable strings
    (e.g. "Total Revenue", "Net Income"), but a handful have alternate
    spellings across companies/periods, so callers pass several candidates.
    """
    if df is None or df.empty:
        return None
    for label in labels:
        if label in df.index:
            s = df.loc[label]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return s.astype(float)
    return None
