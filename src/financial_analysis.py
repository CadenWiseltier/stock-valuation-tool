"""
Historical financial-statement analysis.

Turns the raw statements in a `StockDataBundle` (src/data.py) into the
metrics used throughout the rest of the application: growth rates, margins,
free cash flow, balance-sheet ratios, ROE, and ROIC.

Every function here is a pure calculation on pandas Series/DataFrames --
no network calls happen in this module. Any metric that cannot be computed
because the underlying line item is missing from the data provider is
returned as `np.nan`, which the UI layer (app.py) renders as "N/A". Nothing
in this module invents or estimates a historical figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import (
    StockDataBundle, row, infer_peer_sector,
    REVENUE as _REVENUE, GROSS_PROFIT as _GROSS_PROFIT,
    OPERATING_INCOME as _OPERATING_INCOME, NET_INCOME as _NET_INCOME,
    DILUTED_EPS as _DILUTED_EPS, BASIC_EPS as _BASIC_EPS,
    EBIT as _EBIT, EBITDA as _EBITDA, PRETAX_INCOME as _PRETAX_INCOME,
    TAX_PROVISION as _TAX_PROVISION, INTEREST_EXPENSE as _INTEREST_EXPENSE,
    OPERATING_CASH_FLOW as _OPERATING_CASH_FLOW, CAPEX as _CAPEX,
    D_AND_A_CF as _D_AND_A_CF, CASH as _CASH, TOTAL_DEBT as _TOTAL_DEBT,
    CURRENT_ASSETS as _CURRENT_ASSETS, CURRENT_LIABILITIES as _CURRENT_LIABILITIES,
    TOTAL_ASSETS as _TOTAL_ASSETS, TOTAL_EQUITY as _TOTAL_EQUITY,
    LONG_TERM_DEBT as _LONG_TERM_DEBT, CURRENT_DEBT as _CURRENT_DEBT,
    DILUTED_SHARES as _DILUTED_SHARES,
)


def _pct_change(series: pd.Series) -> pd.Series:
    """Period-over-period percentage growth, with the first period as NaN."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.pct_change()


def _pct_change_from_positive_base(series: pd.Series) -> pd.Series:
    """Period-over-period percentage growth, masked to NaN whenever the
    PRIOR period's value was zero or negative.

    A percent change computed from a non-positive base is not economically
    meaningful -- EPS moving from -$0.27 to $2.90 is not "a -1174% change"
    in any useful sense, since the sign of the denominator flips what
    "growth" even means. Left in, a single such data point is often an
    order of magnitude larger than every other observation in the series
    and silently dominates any downstream statistic computed from it (a
    volatility/std measure especially). This mirrors the "NM" (Not
    Meaningful) convention equity research uses for growth rates computed
    across a loss-to-profit (or profit-to-loss) transition. Used for EPS
    growth and FCF growth, both of which can genuinely cross zero.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    growth = series.pct_change()
    prior = series.shift(1)
    return growth.where(prior > 0)


def _safe_div(numerator: pd.Series | float, denominator: pd.Series | float,
              require_positive_denominator: bool = False):
    """Element-wise division that returns NaN instead of raising/inf on a
    zero or missing denominator.

    `require_positive_denominator=True` additionally masks any result
    where the denominator is zero OR NEGATIVE, not just literally zero.
    This matters for "capital base" denominators (shareholders' equity,
    EBITDA, EBIT) that can legitimately go negative for reasons UNRELATED
    to what the ratio is trying to measure -- e.g. a mature, highly
    profitable company with a history of aggressive share buybacks (found
    in testing: McDonald's, Starbucks) can have NEGATIVE book equity,
    which flips the sign of Net Income / Equity and makes a genuinely
    profitable company's ROE read as a catastrophic loss. Rather than
    report that sign-flipped (and therefore backwards) result, the ratio
    is masked as Not Meaningful, exactly like the loss-to-profit growth
    masking in `_pct_change_from_positive_base` above.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    if require_positive_denominator:
        if isinstance(denominator, pd.Series):
            result = result.where(denominator > 0)
        elif not (denominator is not None and denominator == denominator and denominator > 0):
            return np.nan
    if isinstance(result, pd.Series):
        return result.replace([np.inf, -np.inf], np.nan)
    if result in (np.inf, -np.inf):
        return np.nan
    return result


def is_financial_services_company(bundle: StockDataBundle) -> bool:
    """True if `bundle`'s (override-aware) sector is "Financial Services" --
    i.e. a bank, insurer, broker-dealer, asset manager, or lender.

    For these companies, GAAP operating cash flow includes loan
    origination/sale, deposit, premium, and similar balance-sheet-growth
    activity that is a core, ONGOING part of the business model -- not a
    discretionary or one-time item the way CapEx is for an industrial
    company. This makes "Operating Cash Flow - CapEx" (and OCF/Net Income)
    not meaningful measures for these companies (found in testing: SOFI's
    reported Operating Cash Flow ran to NEGATIVE BILLIONS against ~$3B of
    revenue, swamped by loan origination/deposit flows -- not a sign of
    business distress). Equity research values these companies on ROE,
    Price/Book, P/E, and net interest margin instead of FCF/DCF -- see
    fcf_summary and earnings_to_cash_conversion, and docs/methodology.md.

    Uses the SAME override-aware sector inference as peer selection
    (src/data.py: infer_peer_sector), so a company the data provider
    misclassifies INTO "Financial Services" (e.g. IREN Limited, an AI
    data-center operator) is correctly excluded from this treatment, and
    one misclassified OUT of it would correctly be included.
    """
    return infer_peer_sector(bundle.info) == "Financial Services"


def income_statement_summary(bundle: StockDataBundle) -> pd.DataFrame:
    """Build a per-fiscal-year table of income-statement figures and
    derived margins/growth rates.

    Columns (all indexed by fiscal year-end):
        revenue, gross_profit, operating_income, net_income, diluted_eps,
        revenue_growth, gross_margin, operating_margin, net_margin,
        eps_growth
    """
    df = bundle.income_stmt
    revenue = row(df, *_REVENUE)
    gross_profit = row(df, *_GROSS_PROFIT)
    operating_income = row(df, *_OPERATING_INCOME)
    net_income = row(df, *_NET_INCOME)
    eps = row(df, *_DILUTED_EPS)
    if eps is None:
        eps = row(df, *_BASIC_EPS)

    if revenue is None:
        return pd.DataFrame()

    out = pd.DataFrame({"revenue": revenue})
    out["gross_profit"] = gross_profit
    out["operating_income"] = operating_income
    out["net_income"] = net_income
    out["diluted_eps"] = eps

    out["revenue_growth"] = _pct_change(out["revenue"])
    out["gross_margin"] = _safe_div(out["gross_profit"], out["revenue"])
    out["operating_margin"] = _safe_div(out["operating_income"], out["revenue"])
    out["net_margin"] = _safe_div(out["net_income"], out["revenue"])
    out["eps_growth"] = _pct_change_from_positive_base(out["diluted_eps"])

    out.index.name = "fiscal_year"
    return out


def fcf_summary(bundle: StockDataBundle) -> pd.DataFrame:
    """Build a per-fiscal-year free cash flow table.

    FCF = Operating Cash Flow - CapEx (both from the cash flow statement).
    CapEx is reported by yfinance as a negative number (a cash outflow); we
    take its absolute value so `fcf = ocf - capex_abs` reads naturally.

    For a financial-services company (bank, insurer, broker-dealer, asset
    manager, lender), "free_cash_flow" / "fcf_margin" / "fcf_growth" are
    masked to N/A -- see `is_financial_services_company` for why this
    formula is not meaningful for that class of company. Operating Cash
    Flow and CapEx themselves are still real, reported figures and are
    left in the table for transparency.
    """
    cf = bundle.cash_flow
    inc = bundle.income_stmt
    ocf = row(cf, *_OPERATING_CASH_FLOW)
    capex = row(cf, *_CAPEX)
    revenue = row(inc, *_REVENUE)

    if ocf is None:
        return pd.DataFrame()

    capex_abs = capex.abs() if capex is not None else pd.Series(index=ocf.index, dtype=float)
    out = pd.DataFrame({"operating_cash_flow": ocf})
    out["capex"] = capex_abs.reindex(out.index)

    if is_financial_services_company(bundle):
        out["free_cash_flow"] = np.nan
        out["fcf_margin"] = np.nan
        out["fcf_growth"] = np.nan
    else:
        out["free_cash_flow"] = out["operating_cash_flow"] - out["capex"]
        if revenue is not None:
            out["fcf_margin"] = _safe_div(out["free_cash_flow"], revenue.reindex(out.index))
        else:
            out["fcf_margin"] = np.nan
        out["fcf_growth"] = _pct_change_from_positive_base(out["free_cash_flow"])

    out.index.name = "fiscal_year"
    return out


def earnings_to_cash_conversion(bundle: StockDataBundle) -> pd.Series:
    """Operating Cash Flow / Net Income for each fiscal year.

    A ratio near or above 1.0x suggests reported earnings are backed by
    real cash generation; a ratio well below 1.0x can indicate aggressive
    accrual accounting or deteriorating working capital.

    Masked to N/A for a financial-services company -- see
    `is_financial_services_company`: Operating Cash Flow for a bank or
    lender is dominated by loan origination/deposit activity, not the
    "cash backing up reported earnings" this ratio is meant to capture.
    """
    if is_financial_services_company(bundle):
        return pd.Series(dtype=float)
    ocf = row(bundle.cash_flow, *_OPERATING_CASH_FLOW)
    ni = row(bundle.income_stmt, *_NET_INCOME)
    if ocf is None or ni is None:
        return pd.Series(dtype=float)
    common_idx = ocf.index.intersection(ni.index)
    return _safe_div(ocf.reindex(common_idx), ni.reindex(common_idx))


def balance_sheet_summary(bundle: StockDataBundle) -> pd.DataFrame:
    """Build a per-fiscal-year balance-sheet / financial-health table.

    Columns: cash, total_debt, current_assets, current_liabilities,
    total_assets, total_equity, net_debt, current_ratio, debt_to_equity
    """
    bs = bundle.balance_sheet
    cash = row(bs, *_CASH)
    total_debt = row(bs, *_TOTAL_DEBT)
    current_assets = row(bs, *_CURRENT_ASSETS)
    current_liabilities = row(bs, *_CURRENT_LIABILITIES)
    total_assets = row(bs, *_TOTAL_ASSETS)
    total_equity = row(bs, *_TOTAL_EQUITY)

    if total_assets is None and total_equity is None:
        return pd.DataFrame()

    out = pd.DataFrame(index=bs.columns if not bs.empty else None)
    out["cash"] = cash
    out["total_debt"] = total_debt
    out["current_assets"] = current_assets
    out["current_liabilities"] = current_liabilities
    out["total_assets"] = total_assets
    out["total_equity"] = total_equity
    out["net_debt"] = out["total_debt"] - out["cash"] if total_debt is not None and cash is not None else np.nan
    out["current_ratio"] = _safe_div(out["current_assets"], out["current_liabilities"])
    out["debt_to_equity"] = _safe_div(out["total_debt"], out["total_equity"], require_positive_denominator=True)

    out.index.name = "fiscal_year"
    return out.dropna(how="all")


def credit_metrics(bundle: StockDataBundle) -> dict:
    """Latest-year leverage/coverage metrics that need both the income
    statement and balance sheet: Debt/EBITDA, Net Debt/EBITDA, and interest
    coverage (EBIT / Interest Expense).

    Returns np.nan for any metric whose inputs are unavailable rather than
    guessing -- these are exactly the kind of "silent N/A" cases called out
    in the project's error-handling requirements.

    Debt/EBITDA and Net Debt/EBITDA are additionally masked to N/A when
    EBITDA is zero or negative, and Interest Coverage when EBIT is zero or
    negative. "Leverage relative to earnings" is not a meaningful concept
    when there are no positive earnings to be levered against -- dividing
    debt by a negative EBITDA flips the ratio's sign (found in testing:
    LCID and RIVN, both with deep ongoing operating losses, computed
    NEGATIVE Debt/EBITDA, which this model's scoring scale would read as
    "very low leverage," the opposite of reality). Likewise, taking
    abs(EBIT) for a loss-making company and dividing by interest expense
    produces a large, misleadingly reassuring "coverage" ratio for a
    company that generates no operating profit to cover interest with at
    all (LCID and RIVN both computed 10-27x "coverage" this way despite
    burning billions in operating losses).
    """
    ebitda = row(bundle.income_stmt, *_EBITDA)
    ebit = row(bundle.income_stmt, *_EBIT)
    interest_expense = row(bundle.income_stmt, *_INTEREST_EXPENSE)
    total_debt = row(bundle.balance_sheet, *_TOTAL_DEBT)
    cash = row(bundle.balance_sheet, *_CASH)

    def _latest(s):
        return s.dropna().iloc[-1] if s is not None and not s.dropna().empty else np.nan

    latest_ebitda = _latest(ebitda)
    latest_ebit = _latest(ebit)
    latest_interest = _latest(interest_expense)
    latest_debt = _latest(total_debt)
    latest_cash = _latest(cash)

    net_debt = latest_debt - latest_cash if pd.notna(latest_debt) and pd.notna(latest_cash) else np.nan

    interest_coverage = np.nan
    if pd.notna(latest_ebit) and latest_ebit > 0 and pd.notna(latest_interest):
        interest_coverage = _safe_div(latest_ebit, abs(latest_interest))

    # A company with profitable operations and no meaningful interest expense
    # is not MISSING its interest-coverage data -- it has the strongest
    # possible position on that metric, because there is essentially nothing
    # to cover. Reporting only NaN caused the scoring layer to treat a
    # debt-free balance sheet as an unknown and award neutral half credit,
    # ranking it below a company with real (if comfortably serviced) debt.
    # This flag lets the scorer distinguish "no data" from "no debt"; the
    # ratio itself stays NaN because it is genuinely undefined.
    no_meaningful_interest_expense = bool(
        pd.notna(latest_ebit) and latest_ebit > 0
        and (pd.isna(latest_interest) or abs(latest_interest) < 0.005 * abs(latest_ebit))
    )

    return {
        "debt_to_ebitda": _safe_div(latest_debt, latest_ebitda, require_positive_denominator=True),
        "net_debt_to_ebitda": _safe_div(net_debt, latest_ebitda, require_positive_denominator=True),
        "interest_coverage": interest_coverage,
        "net_debt": net_debt,
        "no_meaningful_interest_expense": no_meaningful_interest_expense,
    }


def roe_series(bundle: StockDataBundle) -> pd.Series:
    """Return on Equity, using AVERAGE shareholders' equity where two
    consecutive periods are available (standard practice, since equity is a
    balance-sheet snapshot but net income is a flow over the period):

        ROE = Net Income / Average Shareholders' Equity
        Average Shareholders' Equity = (Equity_t + Equity_t-1) / 2

    Falls back to period-end equity for the earliest year in the dataset,
    where no prior-year balance is available.

    Masked to N/A for any period where average equity is zero or
    NEGATIVE. A mature, highly profitable company with a history of
    aggressive share buybacks can have negative book equity (found in
    testing: McDonald's, Starbucks) -- dividing a genuinely positive net
    income by negative equity flips the sign and makes a profitable
    company's ROE read as a catastrophic loss (McDonald's computed as low
    as -307% ROE despite ~20% ROIC in the same year). ROE is simply not a
    meaningful ratio when the equity base is negative; standard financial
    data providers report this case as "N/A"/"NM" for the same reason.
    """
    ni = row(bundle.income_stmt, *_NET_INCOME)
    equity = row(bundle.balance_sheet, *_TOTAL_EQUITY)
    if ni is None or equity is None:
        return pd.Series(dtype=float)

    avg_equity = equity.rolling(window=2).mean()
    avg_equity = avg_equity.fillna(equity)  # first period: use period-end equity
    common_idx = ni.index.intersection(avg_equity.index)
    return _safe_div(ni.reindex(common_idx), avg_equity.reindex(common_idx), require_positive_denominator=True)


def roic_series(bundle: StockDataBundle) -> pd.Series:
    """Return on Invested Capital.

        NOPAT (Net Operating Profit After Tax) = EBIT x (1 - Effective Tax Rate)
        Effective Tax Rate = Tax Provision / Pretax Income  (per fiscal year)
        Invested Capital = Total Debt + Total Equity - Cash & Equivalents

        ROIC = NOPAT / Average Invested Capital

    This is the standard "operating" definition of ROIC used in equity
    research: it excludes the effect of a company's financing mix (unlike
    ROE) and excludes excess cash from the capital base, since idle cash
    does not require a return to justify its presence on the balance sheet.
    Average invested capital (current + prior year, when available) is
    used for the same flow-vs-stock reasoning as ROE above.
    """
    ebit = row(bundle.income_stmt, *_EBIT)
    pretax = row(bundle.income_stmt, *_PRETAX_INCOME)
    tax = row(bundle.income_stmt, *_TAX_PROVISION)
    total_debt = row(bundle.balance_sheet, *_TOTAL_DEBT)
    equity = row(bundle.balance_sheet, *_TOTAL_EQUITY)
    cash = row(bundle.balance_sheet, *_CASH)

    if ebit is None or equity is None:
        return pd.Series(dtype=float)

    effective_tax_rate = _safe_div(tax, pretax) if (tax is not None and pretax is not None) else pd.Series(
        0.21, index=ebit.index)
    # Clip to a sane range: extraordinary items can push a single year's
    # effective tax rate negative or above 100%, which would distort NOPAT.
    effective_tax_rate = effective_tax_rate.clip(lower=0.0, upper=0.50) if isinstance(
        effective_tax_rate, pd.Series) else 0.21
    effective_tax_rate = effective_tax_rate.reindex(ebit.index).fillna(0.21)

    nopat = ebit * (1 - effective_tax_rate)

    debt_component = total_debt if total_debt is not None else pd.Series(0.0, index=equity.index)
    cash_component = cash if cash is not None else pd.Series(0.0, index=equity.index)
    invested_capital = (
        debt_component.reindex(equity.index).fillna(0.0)
        + equity
        - cash_component.reindex(equity.index).fillna(0.0)
    )
    avg_invested_capital = invested_capital.rolling(window=2).mean().fillna(invested_capital)

    # Masked to N/A when average invested capital is zero or negative --
    # same reasoning as ROE above (a negative capital base, possible when
    # equity is deeply negative from buybacks and not fully offset by
    # debt, flips the sign of NOPAT / Invested Capital and would make a
    # profitable company's ROIC read as a loss, or vice versa).
    common_idx = nopat.index.intersection(avg_invested_capital.index)
    return _safe_div(nopat.reindex(common_idx), avg_invested_capital.reindex(common_idx),
                      require_positive_denominator=True)


def generate_trend_insights(income_df: pd.DataFrame, fcf_df: pd.DataFrame) -> list[str]:
    """Produce short, data-driven observations about historical trends.

    Every insight below is gated on an actual computed condition -- nothing
    is emitted unless the underlying numbers support it, per the project
    requirement not to make unsupported claims.
    """
    insights: list[str] = []
    if income_df is None or income_df.empty:
        return insights

    rev_growth = income_df["revenue_growth"].dropna()
    op_margin = income_df["operating_margin"].dropna()
    net_margin = income_df["net_margin"].dropna()

    if len(rev_growth) >= 2:
        recent_growth = rev_growth.iloc[-1]
        avg_growth = rev_growth.iloc[:-1].mean()
        if recent_growth < avg_growth - 0.03:
            insights.append(
                f"Revenue growth decelerated to {recent_growth:.1%} in the most recent "
                f"fiscal year versus a prior average of {avg_growth:.1%}."
            )
        elif recent_growth > avg_growth + 0.03:
            insights.append(
                f"Revenue growth accelerated to {recent_growth:.1%} in the most recent "
                f"fiscal year versus a prior average of {avg_growth:.1%}."
            )

    if len(op_margin) >= 2 and len(rev_growth) >= 1:
        margin_trend = op_margin.iloc[-1] - op_margin.iloc[0]
        if rev_growth.iloc[-1] > 0 and margin_trend < -0.02:
            insights.append(
                "Revenue is growing but operating margin has declined "
                f"{abs(margin_trend):.1%} over the period -- a potential cost-discipline concern."
            )
        elif margin_trend > 0.02:
            insights.append(
                f"Operating margin has expanded {margin_trend:.1%} over the period, "
                "indicating improving operating leverage or cost control."
            )

    if len(net_margin) >= 2:
        net_trend = net_margin.iloc[-1] - net_margin.iloc[0]
        if net_trend < -0.02:
            insights.append(
                f"Net margin has contracted {abs(net_trend):.1%} over the period."
            )

    if fcf_df is not None and not fcf_df.empty:
        fcf = fcf_df["free_cash_flow"].dropna()
        if len(fcf) >= 2:
            if (fcf > 0).all():
                insights.append("Free cash flow has been positive in every reported fiscal year.")
            elif (fcf < 0).any():
                insights.append("Free cash flow was negative in at least one reported fiscal year.")

    return insights


def historical_pe_series(bundle: StockDataBundle, max_date_tolerance_days: int = 10) -> pd.Series:
    """Trailing P/E ratio at each historical fiscal year-end: the stock's
    price on (or nearest to) that date, divided by that fiscal year's
    reported diluted EPS.

    This exists to answer "is the stock cheap or expensive relative to its
    OWN valuation history?" -- a company-specific reference point that a
    peer comparison alone cannot provide. A fiscal year with non-positive
    EPS is excluded (a negative P/E is not a meaningful valuation
    reference). Requires split/dividend-adjusted price history (see the
    note in src/data.py: fetch_stock_data on why auto_adjust=True is used)
    so that a historical stock split does not throw the ratio off by the
    split factor.
    """
    eps = row(bundle.income_stmt, *_DILUTED_EPS)
    if eps is None or bundle.price_history is None or bundle.price_history.empty:
        return pd.Series(dtype=float)
    if "Close" not in bundle.price_history.columns:
        return pd.Series(dtype=float)

    close = bundle.price_history["Close"]
    # yfinance's intraday/daily price index is timezone-aware (localized to
    # the exchange timezone); financial-statement period dates are
    # timezone-naive. Normalize to naive on both sides before comparing --
    # only the calendar date matters for a +/-10-day nearest-match, not the
    # time-of-day or timezone.
    if close.index.tz is not None:
        close = close.copy()
        close.index = close.index.tz_localize(None)
    tolerance = pd.Timedelta(days=max_date_tolerance_days)

    values, index = [], []
    for fiscal_date, eps_value in eps.items():
        fiscal_date = pd.Timestamp(fiscal_date)
        if fiscal_date.tzinfo is not None:
            fiscal_date = fiscal_date.tz_localize(None)
        if pd.isna(eps_value) or eps_value <= 0:
            continue
        # Find the nearest trading day to the fiscal year-end within tolerance.
        pos = close.index.searchsorted(fiscal_date)
        candidates = close.index[max(pos - 1, 0):pos + 2]
        if len(candidates) == 0:
            continue
        nearest = min(candidates, key=lambda d: abs(d - fiscal_date))
        if abs(nearest - fiscal_date) > tolerance:
            continue
        price_at_date = close.loc[nearest]
        values.append(price_at_date / eps_value)
        index.append(fiscal_date)

    return pd.Series(values, index=index, dtype=float)


def diluted_shares_growth_rate(bundle: StockDataBundle) -> float:
    """Compound annual growth rate of diluted weighted-average shares
    outstanding across the available historical income statements.

    A positive rate indicates the share count is growing (dilution from
    stock-based compensation or equity issuance, which reduces each
    existing share's claim on future cash flows); a negative rate
    indicates net share buybacks (accretive to remaining shareholders).
    Used as a deduction in the expected-return calculation (src/scoring.py)
    since dilution is a real, measurable drag on a per-share return that
    a pure earnings/FCF growth figure does not capture.

    Clipped to +/-30% per year: an early-stage or rapidly-growing company
    raising heavy equity capital during a temporary build-out phase (e.g.
    a data-center operator funding new capacity) can show a historical
    diluted-share CAGR of 70%+ -- extrapolating that indefinitely into a
    forward-looking expected-return calculation is not defensible (that
    pace of issuance cannot continue for long by construction), so it is
    capped at a still-severe but bounded rate.
    """
    shares = row(bundle.income_stmt, *_DILUTED_SHARES)
    if shares is None:
        return np.nan
    s = shares.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return np.nan
    n_periods = len(s) - 1
    rate = (s.iloc[-1] / s.iloc[0]) ** (1 / n_periods) - 1
    return float(np.clip(rate, -0.30, 0.30))
