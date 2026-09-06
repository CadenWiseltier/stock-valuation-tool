"""
Discounted Cash Flow (DCF) valuation engine.

Implements a standard unlevered Free-Cash-Flow-to-Firm (FCFF) DCF:

    1. Generate baseline forecast assumptions automatically from historical
       financials (see `build_baseline_assumptions`).
    2. Project revenue, EBIT, NOPAT, and FCFF for the forecast horizon.
    3. Discount each year's FCFF to present value using WACC.
    4. Estimate a terminal value (Gordon Growth, with an Exit Multiple
       cross-check) and discount it to present value.
    5. Sum to Enterprise Value, bridge to Equity Value, and divide by
       diluted shares outstanding for an intrinsic value per share.

Every assumption used is returned alongside the result and tagged as one of:
    "historical"           - an observed, reported figure
    "model_assumption"     - derived from historical data by a documented
                              rule (e.g. fading growth toward a long-run rate)
    "external_assumption"  - a widely-used macro/market input not derived
                              from the company's own filings (e.g. the
                              equity risk premium)

so the UI can clearly separate fact from estimate, per the project's
transparency requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data import (
    StockDataBundle, row,
    EBIT as _EBIT, PRETAX_INCOME as _PRETAX_INCOME, REVENUE as _REVENUE,
    TAX_PROVISION as _TAX_PROVISION, TOTAL_DEBT as _TOTAL_DEBT,
    TOTAL_EQUITY as _TOTAL_EQUITY, CASH as _CASH, CAPEX as _CAPEX,
    D_AND_A_CF as _D_AND_A_CF,
)

# ---------------------------------------------------------------------------
# External-assumption defaults.
#
# These are not derived from the target company's own filings, so they are
# always labeled "external_assumption" and are the primary levers exposed in
# the Advanced Assumptions panel. Values are reasonable, commonly cited
# long-run figures rather than a live macro feed (documented in
# docs/methodology.md); an advanced user can override either.
# ---------------------------------------------------------------------------
DEFAULT_RISK_FREE_RATE = 0.042        # ~10-year U.S. Treasury yield, long-run reference
DEFAULT_EQUITY_RISK_PREMIUM = 0.045   # Widely cited long-run U.S. equity risk premium
DEFAULT_TERMINAL_GROWTH = 0.025       # ~long-run nominal GDP growth proxy
DEFAULT_TAX_RATE = 0.21               # U.S. federal statutory corporate tax rate
FORECAST_YEARS = 5

# Minimum spread enforced between WACC and terminal growth before computing
# the Gordon Growth terminal value (see run_dcf). A pure "wacc > g" check
# is not enough on its own: as the spread shrinks toward zero the terminal
# value explodes even though the formula stays mathematically defined, so a
# small positive floor keeps the model numerically stable.
MIN_WACC_TERMINAL_GROWTH_SPREAD = 0.01

# Sanity ceiling on the automatic CapEx-% and D&A-%-of-revenue assumptions
# (see build_baseline_assumptions). A young, capital-intensive company mid
# build-out (e.g. a data-center operator still ramping revenue) can show a
# historical CapEx/revenue ratio of 200%+ in a single year -- averaging
# that across all available history and projecting it flat for 5 years
# produces an economically absurd forecast (found in testing: -$2B+/year
# of projected FCF against roughly $1B of revenue). Even the most
# capital-intensive mature businesses rarely sustain CapEx above ~50% of
# revenue in steady state, so this caps the ASSUMPTION (not the company's
# real historical ratio, which is still shown to the user) at that level.
MAX_CAPEX_OR_DA_PCT_OF_REVENUE = 0.50


@dataclass
class Assumption:
    """A single labeled model input, shown verbatim in the UI."""
    value: float
    kind: str  # "historical" | "model_assumption" | "external_assumption"
    note: str = ""


@dataclass
class DCFAssumptions:
    revenue_growth_path: list[float]          # one growth rate per forecast year
    operating_margin_path: list[float]
    tax_rate: Assumption
    da_pct_revenue: Assumption
    capex_pct_revenue: Assumption
    nwc_pct_revenue_change: Assumption
    risk_free_rate: Assumption
    equity_risk_premium: Assumption
    beta: Assumption
    cost_of_debt: Assumption
    terminal_growth: Assumption
    forecast_years: int = FORECAST_YEARS


@dataclass
class WACCResult:
    cost_of_equity: float
    after_tax_cost_of_debt: float
    pretax_cost_of_debt: float
    equity_value_weight: float
    debt_value_weight: float
    market_cap: float
    total_debt: float
    wacc: float


@dataclass
class DCFResult:
    assumptions: DCFAssumptions
    wacc: WACCResult
    projection: pd.DataFrame           # year-by-year forecast table
    terminal_value_gordon: float
    terminal_value_exit_multiple: Optional[float]
    pv_forecast_fcf: float
    pv_terminal_value: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    shares_outstanding: float
    intrinsic_value_per_share: float
    current_price: float
    upside: float


def _latest(series: Optional[pd.Series]) -> float:
    if series is None:
        return np.nan
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def _cagr(series: Optional[pd.Series]) -> float:
    """Compound annual growth rate between the first and last available
    values of a historical series.

    Requires BOTH endpoints to be positive, not just the starting value:
    a CAGR computed from a negative ending value (e.g. a series that went
    from positive to negative, such as EBIT for a company that started
    profitable and later posted a loss) requires raising a negative number
    to a fractional power, which is undefined over the reals and -- left
    unguarded -- silently returns NaN with a RuntimeWarning rather than a
    clean, intentional "not computable" result.
    """
    if series is None:
        return np.nan
    s = series.dropna()
    if len(s) < 2 or s.iloc[0] <= 0 or s.iloc[-1] <= 0:
        return np.nan
    n_periods = len(s) - 1
    return (s.iloc[-1] / s.iloc[0]) ** (1 / n_periods) - 1


def calculate_wacc(
    bundle: StockDataBundle,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    beta_override: Optional[float] = None,
    cost_of_debt_override: Optional[float] = None,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> WACCResult:
    """Weighted Average Cost of Capital.

        Cost of Equity (CAPM) = Risk-Free Rate + Beta x Equity Risk Premium
        Pre-Tax Cost of Debt  = Interest Expense / Total Debt  (proxy, when
                                 available), else a credit-spread default
        After-Tax Cost of Debt = Pre-Tax Cost of Debt x (1 - Tax Rate)

        WACC = E/(D+E) x Cost of Equity + D/(D+E) x After-Tax Cost of Debt

    E is the company's market capitalization (market value of equity) and D
    is total debt from the latest balance sheet (a common simplifying proxy
    for the market value of debt, since most corporate debt is not
    continuously marked to market).
    """
    info = bundle.info
    market_cap = info.get("marketCap")
    market_cap = float(market_cap) if market_cap is not None else np.nan
    if np.isnan(market_cap):
        # Step 1 fallback: calculate from other available data (price x
        # shares) rather than assuming zero equity value, which would
        # force debt_weight to 100% in the WACC blend below and ignore
        # cost of equity entirely.
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares = info.get("sharesOutstanding")
        if price is not None and shares is not None:
            market_cap = float(price) * float(shares)
    beta = beta_override if beta_override is not None else info.get("beta")
    beta = beta if beta is not None else 1.0  # market-average default if missing

    total_debt = _latest(row(bundle.balance_sheet, *_TOTAL_DEBT))
    if np.isnan(total_debt):
        # Step 3 fallback: try the lightweight `info` field (a different
        # endpoint from the balance-sheet statement) before assuming the
        # company is debt-free, which a genuinely missing value is not.
        info_total_debt = info.get("totalDebt")
        total_debt = float(info_total_debt) if info_total_debt is not None else 0.0

    if cost_of_debt_override is not None:
        pretax_cost_of_debt = cost_of_debt_override
    else:
        interest_expense = _latest(row(bundle.income_stmt, "Interest Expense", "Interest Expense Non Operating"))
        if not np.isnan(interest_expense) and total_debt > 0:
            pretax_cost_of_debt = min(abs(interest_expense) / total_debt, 0.15)
        else:
            # No reliable interest-expense/debt data: fall back to a
            # risk-free-rate-plus-spread proxy typical of an investment
            # grade issuer.
            pretax_cost_of_debt = risk_free_rate + 0.015

    cost_of_equity = risk_free_rate + beta * equity_risk_premium
    after_tax_cost_of_debt = pretax_cost_of_debt * (1 - tax_rate)

    total_capital = (market_cap if not np.isnan(market_cap) else 0.0) + total_debt
    if total_capital <= 0:
        equity_weight, debt_weight = 1.0, 0.0
    else:
        equity_weight = (market_cap if not np.isnan(market_cap) else 0.0) / total_capital
        debt_weight = total_debt / total_capital

    wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt

    return WACCResult(
        cost_of_equity=cost_of_equity,
        after_tax_cost_of_debt=after_tax_cost_of_debt,
        pretax_cost_of_debt=pretax_cost_of_debt,
        equity_value_weight=equity_weight,
        debt_value_weight=debt_weight,
        market_cap=market_cap,
        total_debt=total_debt,
        wacc=wacc,
    )


def build_baseline_assumptions(
    bundle: StockDataBundle,
    forecast_years: int = FORECAST_YEARS,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
) -> DCFAssumptions:
    """Derive automatic DCF assumptions from historical data.

    Revenue growth: starts at the company's historical revenue CAGR (capped
    to a sane range) and fades linearly toward the terminal growth rate by
    the final forecast year -- a standard "fade to a normalized long-run
    rate" approach that avoids extrapolating a high growth rate forever.

    Operating margin: held at the historical average of the last available
    fiscal years (a simple, transparent normalization -- no attempt to
    project margin expansion or contraction, since that would be
    speculative beyond what history supports).

    Tax rate: historical average effective tax rate when computable,
    otherwise the U.S. statutory default.

    D&A and CapEx: each estimated as a percentage of revenue, using the
    historical average of that ratio (a standard simplifying approach when
    a detailed PP&E schedule is not available).

    Working capital: modeled as a constant percentage-of-revenue change,
    using the historical average year-over-year change in (current assets -
    current liabilities) as a fraction of the change in revenue -- kept
    simple and clearly labeled as an assumption.
    """
    revenue = row(bundle.income_stmt, *_REVENUE)
    ebit = row(bundle.income_stmt, *_EBIT)
    pretax = row(bundle.income_stmt, *_PRETAX_INCOME)
    tax = row(bundle.income_stmt, *_TAX_PROVISION)
    capex = row(bundle.cash_flow, *_CAPEX)
    da = row(bundle.cash_flow, *_D_AND_A_CF)

    # --- Revenue growth path ---
    hist_cagr = _cagr(revenue)
    if np.isnan(hist_cagr):
        starting_growth = 0.05  # neutral default when no history exists
    else:
        starting_growth = float(np.clip(hist_cagr, -0.10, 0.30))
    growth_path = list(np.linspace(starting_growth, terminal_growth, forecast_years))

    # --- Operating margin path ---
    # Recency-weighted average, not a flat mean: a young or rapidly-
    # maturing company's oldest reported margins can be wildly
    # unrepresentative of its current run-rate (e.g. a company moving from
    # a heavy loss-making build-out phase to recent profitability -- found
    # in testing, a flat historical mean anchored the ENTIRE 5-year
    # forecast to a deeply negative margin despite a clear, real recent
    # improvement trend). Linearly increasing weights (oldest year weight
    # 1, newest year weight n) keep the forecast grounded in multiple years
    # of real data while anchoring it closer to the business's current
    # state than either a stale multi-year blend or a single
    # (potentially anomalous) most-recent year would.
    if revenue is not None and ebit is not None:
        common_idx = revenue.index.intersection(ebit.index)
        hist_margin = (ebit.reindex(common_idx) / revenue.reindex(common_idx)).dropna()
        if not hist_margin.empty:
            weights = np.arange(1, len(hist_margin) + 1)
            avg_margin = float(np.average(hist_margin.values, weights=weights))
        else:
            avg_margin = 0.10
    else:
        avg_margin = 0.10
    margin_path = [avg_margin] * forecast_years

    # --- Tax rate ---
    if tax is not None and pretax is not None:
        common_idx = tax.index.intersection(pretax.index)
        eff_rate = (tax.reindex(common_idx) / pretax.reindex(common_idx)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        eff_rate = eff_rate[(eff_rate >= 0) & (eff_rate <= 0.5)]
        tax_rate_val = float(eff_rate.mean()) if not eff_rate.empty else DEFAULT_TAX_RATE
        tax_kind, tax_note = "historical", "Historical average effective tax rate"
    else:
        tax_rate_val, tax_kind, tax_note = DEFAULT_TAX_RATE, "external_assumption", "U.S. statutory default (data unavailable)"

    # --- D&A % of revenue ---
    if da is not None and revenue is not None:
        common_idx = da.index.intersection(revenue.index)
        ratio = (da.reindex(common_idx).abs() / revenue.reindex(common_idx)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            da_pct_raw = float(ratio.mean())
            da_pct = min(da_pct_raw, MAX_CAPEX_OR_DA_PCT_OF_REVENUE)
            if da_pct_raw > MAX_CAPEX_OR_DA_PCT_OF_REVENUE:
                da_kind, da_note = "historical", (
                    f"Historical average D&A/revenue ({da_pct_raw:.0%}) capped at "
                    f"{MAX_CAPEX_OR_DA_PCT_OF_REVENUE:.0%} -- see docs/methodology.md limitations"
                )
            else:
                da_kind, da_note = "historical", "Historical average D&A / revenue"
        else:
            da_pct, da_kind, da_note = 0.03, "model_assumption", "Default D&A / revenue (data unavailable)"
    else:
        da_pct, da_kind, da_note = 0.03, "model_assumption", "Default D&A / revenue (data unavailable)"

    # --- CapEx % of revenue ---
    if capex is not None and revenue is not None:
        common_idx = capex.index.intersection(revenue.index)
        ratio = (capex.reindex(common_idx).abs() / revenue.reindex(common_idx)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            capex_pct_raw = float(ratio.mean())
            capex_pct = min(capex_pct_raw, MAX_CAPEX_OR_DA_PCT_OF_REVENUE)
            if capex_pct_raw > MAX_CAPEX_OR_DA_PCT_OF_REVENUE:
                capex_kind, capex_note = "historical", (
                    f"Historical average CapEx/revenue ({capex_pct_raw:.0%}) capped at "
                    f"{MAX_CAPEX_OR_DA_PCT_OF_REVENUE:.0%} -- see docs/methodology.md limitations"
                )
            else:
                capex_kind, capex_note = "historical", "Historical average CapEx / revenue"
        else:
            capex_pct, capex_kind, capex_note = 0.04, "model_assumption", "Default CapEx / revenue (data unavailable)"
    else:
        capex_pct, capex_kind, capex_note = 0.04, "model_assumption", "Default CapEx / revenue (data unavailable)"

    # --- Working capital: modeled as a fraction of revenue growth ---
    nwc_pct = 0.05  # each $1 of incremental revenue requires ~5c of incremental net working capital
    nwc_note = "Model assumption: incremental NWC investment ~5% of incremental revenue (industry-typical default)"

    beta = bundle.info.get("beta")
    beta_val = float(beta) if beta is not None else 1.0
    beta_kind = "historical" if beta is not None else "external_assumption"
    beta_note = "Data provider 5-year monthly beta" if beta is not None else "Market-average default (beta unavailable)"

    return DCFAssumptions(
        revenue_growth_path=growth_path,
        operating_margin_path=margin_path,
        tax_rate=Assumption(tax_rate_val, tax_kind, tax_note),
        da_pct_revenue=Assumption(da_pct, da_kind, da_note),
        capex_pct_revenue=Assumption(capex_pct, capex_kind, capex_note),
        nwc_pct_revenue_change=Assumption(nwc_pct, "model_assumption", nwc_note),
        risk_free_rate=Assumption(risk_free_rate, "external_assumption", "Long-run reference risk-free rate"),
        equity_risk_premium=Assumption(equity_risk_premium, "external_assumption", "Long-run U.S. equity risk premium"),
        beta=Assumption(beta_val, beta_kind, beta_note),
        cost_of_debt=Assumption(np.nan, "model_assumption", "Derived within WACC calculation"),
        terminal_growth=Assumption(terminal_growth, "external_assumption", "Long-run nominal GDP growth proxy"),
        forecast_years=forecast_years,
    )


def run_dcf(
    bundle: StockDataBundle,
    assumptions: DCFAssumptions,
    wacc_result: WACCResult,
    exit_multiple: Optional[float] = None,
) -> DCFResult:
    """Project FCFF, discount it, and back out an intrinsic share value.

        FCFF = EBIT x (1 - Tax Rate) + D&A - CapEx - Change in NWC
        PV(FCFF_t) = FCFF_t / (1 + WACC)^t

        Terminal Value (Gordon Growth) = FCFF_(n+1) / (WACC - g)
        PV(Terminal Value) = Terminal Value / (1 + WACC)^n

        Enterprise Value = sum(PV(FCFF)) + PV(Terminal Value)
        Equity Value = Enterprise Value - Net Debt
        Intrinsic Value / Share = Equity Value / Diluted Shares Outstanding
    """
    revenue_series = row(bundle.income_stmt, *_REVENUE)
    last_revenue = _latest(revenue_series)
    if np.isnan(last_revenue):
        last_revenue = bundle.info.get("totalRevenue", np.nan)

    n = assumptions.forecast_years
    wacc = wacc_result.wacc
    g = assumptions.terminal_growth.value

    years = list(range(1, n + 1))
    revenues, ebits, nopats, das, capexs, nwc_changes, fcffs, pvs = ([] for _ in range(8))

    revenue = last_revenue
    prev_revenue = last_revenue
    for i in range(n):
        growth = assumptions.revenue_growth_path[i]
        revenue = prev_revenue * (1 + growth)
        margin = assumptions.operating_margin_path[i]
        ebit = revenue * margin
        nopat = ebit * (1 - assumptions.tax_rate.value)
        da = revenue * assumptions.da_pct_revenue.value
        capex = revenue * assumptions.capex_pct_revenue.value
        delta_revenue = revenue - prev_revenue
        nwc_change = delta_revenue * assumptions.nwc_pct_revenue_change.value
        fcff = nopat + da - capex - nwc_change
        pv = fcff / ((1 + wacc) ** (i + 1))

        revenues.append(revenue)
        ebits.append(ebit)
        nopats.append(nopat)
        das.append(da)
        capexs.append(capex)
        nwc_changes.append(nwc_change)
        fcffs.append(fcff)
        pvs.append(pv)
        prev_revenue = revenue

    projection = pd.DataFrame({
        "year": years,
        "revenue": revenues,
        "revenue_growth": assumptions.revenue_growth_path,
        "ebit": ebits,
        "operating_margin": assumptions.operating_margin_path,
        "tax_rate": [assumptions.tax_rate.value] * n,
        "nopat": nopats,
        "d_and_a": das,
        "capex": capexs,
        "change_in_nwc": nwc_changes,
        "free_cash_flow": fcffs,
        "present_value_fcf": pvs,
    }).set_index("year")

    pv_forecast_fcf = float(np.sum(pvs))

    # Terminal value requires WACC > g, otherwise the Gordon Growth formula
    # is mathematically invalid (division by zero or negative denominator).
    # Beyond that hard requirement, the formula is also numerically unstable
    # whenever WACC and g are merely CLOSE together: dividing by a small
    # (WACC - g) amplifies terminal value enormously for a tiny change in
    # either input. This matters in practice because the Bull scenario
    # (src/scenarios.py) simultaneously lowers WACC and raises terminal
    # growth -- for a company whose base-case spread is already thin, that
    # combination can shrink the denominator toward zero and produce an
    # economically implausible terminal value (seen in testing: a "Bull
    # case" intrinsic value dozens of times the current share price). A
    # minimum spread is a standard practitioner guardrail against exactly
    # this instability, not just the literal wacc<=g edge case.
    spread = wacc - g
    if spread < MIN_WACC_TERMINAL_GROWTH_SPREAD:
        g_effective = wacc - MIN_WACC_TERMINAL_GROWTH_SPREAD
    else:
        g_effective = g

    fcff_terminal_year = fcffs[-1]
    fcff_next = fcff_terminal_year * (1 + g_effective)
    terminal_value_gordon = fcff_next / (wacc - g_effective)
    pv_terminal_value = terminal_value_gordon / ((1 + wacc) ** n)

    terminal_value_exit_multiple = None
    if exit_multiple is not None:
        ebitda_terminal = ebits[-1] + das[-1]
        terminal_value_exit_multiple = ebitda_terminal * exit_multiple

    enterprise_value = pv_forecast_fcf + pv_terminal_value

    total_debt = wacc_result.total_debt
    cash = _latest(row(bundle.balance_sheet, *_CASH))
    if np.isnan(cash):
        # Step 3 fallback: try the lightweight `info` field before assuming
        # zero cash, which would OVERSTATE net debt (understating equity
        # value) for any company whose balance-sheet cash row is missing.
        info_cash = bundle.info.get("totalCash")
        cash = float(info_cash) if info_cash is not None else 0.0
    net_debt = total_debt - cash

    equity_value = enterprise_value - net_debt

    shares_outstanding = bundle.info.get("sharesOutstanding") or np.nan
    current_price = bundle.info.get("currentPrice") or bundle.info.get("regularMarketPrice") or np.nan

    if shares_outstanding and not np.isnan(shares_outstanding) and shares_outstanding > 0:
        intrinsic_value_per_share = equity_value / shares_outstanding
    else:
        intrinsic_value_per_share = np.nan

    if not np.isnan(intrinsic_value_per_share) and current_price and not np.isnan(current_price) and current_price > 0:
        upside = (intrinsic_value_per_share / current_price) - 1
    else:
        upside = np.nan

    return DCFResult(
        assumptions=assumptions,
        wacc=wacc_result,
        projection=projection,
        terminal_value_gordon=terminal_value_gordon,
        terminal_value_exit_multiple=terminal_value_exit_multiple,
        pv_forecast_fcf=pv_forecast_fcf,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
        intrinsic_value_per_share=intrinsic_value_per_share,
        current_price=current_price,
        upside=upside,
    )
