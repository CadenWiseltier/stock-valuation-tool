"""Unit tests for src/dcf.py: CAPM, WACC, FCFF projection, terminal value,
and the enterprise-value-to-intrinsic-value-per-share bridge."""

import numpy as np
import pandas as pd
import pytest

from src.data import StockDataBundle
from src.dcf import (
    Assumption, DCFAssumptions, MAX_CAPEX_OR_DA_PCT_OF_REVENUE,
    build_baseline_assumptions, calculate_wacc, run_dcf,
)


def test_capm_cost_of_equity(synthetic_bundle):
    """Cost of Equity = Risk-Free Rate + Beta x Equity Risk Premium."""
    result = calculate_wacc(synthetic_bundle, risk_free_rate=0.04, equity_risk_premium=0.05)
    expected_cost_of_equity = 0.04 + synthetic_bundle.info["beta"] * 0.05
    assert result.cost_of_equity == pytest.approx(expected_cost_of_equity)


def test_wacc_weights_sum_to_one(synthetic_bundle):
    result = calculate_wacc(synthetic_bundle)
    assert result.equity_value_weight + result.debt_value_weight == pytest.approx(1.0)


def test_wacc_formula(synthetic_bundle):
    """WACC = E/(D+E) x Cost of Equity + D/(D+E) x After-Tax Cost of Debt."""
    result = calculate_wacc(synthetic_bundle, risk_free_rate=0.042, equity_risk_premium=0.045, tax_rate=0.21)
    expected = (
        result.equity_value_weight * result.cost_of_equity
        + result.debt_value_weight * result.after_tax_cost_of_debt
    )
    assert result.wacc == pytest.approx(expected)

    # Sanity check against a fully independent hand calculation using the
    # fixture's known inputs: market cap 10,000, total debt 300, beta 1.2,
    # interest expense 10 (latest year) -> pretax cost of debt = 10/300.
    cost_of_equity = 0.042 + 1.2 * 0.045
    pretax_cod = 10.0 / 300.0
    after_tax_cod = pretax_cod * (1 - 0.21)
    equity_weight = 10_000 / (10_000 + 300)
    debt_weight = 300 / (10_000 + 300)
    expected_wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cod
    assert result.wacc == pytest.approx(expected_wacc, rel=1e-6)


def _single_year_assumptions(terminal_growth=0.025) -> DCFAssumptions:
    """A minimal one-year forecast so the DCF math can be checked by hand."""
    return DCFAssumptions(
        revenue_growth_path=[0.10],
        operating_margin_path=[0.20],
        tax_rate=Assumption(0.21, "external_assumption"),
        da_pct_revenue=Assumption(0.05, "model_assumption"),
        capex_pct_revenue=Assumption(0.06, "model_assumption"),
        nwc_pct_revenue_change=Assumption(0.05, "model_assumption"),
        risk_free_rate=Assumption(0.042, "external_assumption"),
        equity_risk_premium=Assumption(0.045, "external_assumption"),
        beta=Assumption(1.2, "historical"),
        cost_of_debt=Assumption(np.nan, "model_assumption"),
        terminal_growth=Assumption(terminal_growth, "external_assumption"),
        forecast_years=1,
    )


def test_fcff_formula(synthetic_bundle):
    """FCFF = EBIT x (1 - Tax Rate) + D&A - CapEx - Change in NWC."""
    assumptions = _single_year_assumptions()
    wacc_result = calculate_wacc(synthetic_bundle, risk_free_rate=0.042, equity_risk_premium=0.045)
    result = run_dcf(synthetic_bundle, assumptions, wacc_result)

    last_revenue = 1210.0  # latest historical revenue in the fixture
    revenue_yr1 = last_revenue * 1.10
    ebit = revenue_yr1 * 0.20
    nopat = ebit * (1 - 0.21)
    da = revenue_yr1 * 0.05
    capex = revenue_yr1 * 0.06
    nwc_change = (revenue_yr1 - last_revenue) * 0.05
    expected_fcff = nopat + da - capex - nwc_change

    assert result.projection["free_cash_flow"].iloc[0] == pytest.approx(expected_fcff, rel=1e-6)


def test_terminal_value_and_intrinsic_value(synthetic_bundle):
    """Terminal Value = FCFF_(n+1) / (WACC - g); Equity Value = EV - Net Debt;
    Intrinsic Value / Share = Equity Value / Shares Outstanding."""
    assumptions = _single_year_assumptions()
    wacc_result = calculate_wacc(synthetic_bundle, risk_free_rate=0.042, equity_risk_premium=0.045)
    result = run_dcf(synthetic_bundle, assumptions, wacc_result)

    fcff = result.projection["free_cash_flow"].iloc[0]
    g = assumptions.terminal_growth.value
    wacc = wacc_result.wacc

    expected_tv = (fcff * (1 + g)) / (wacc - g)
    expected_pv_tv = expected_tv / (1 + wacc)
    expected_pv_fcf = fcff / (1 + wacc)
    expected_ev = expected_pv_fcf + expected_pv_tv

    assert result.terminal_value_gordon == pytest.approx(expected_tv, rel=1e-6)
    assert result.pv_terminal_value == pytest.approx(expected_pv_tv, rel=1e-6)
    assert result.enterprise_value == pytest.approx(expected_ev, rel=1e-6)

    net_debt = 300.0 - 300.0  # total debt - cash, both from the latest fixture year
    expected_equity_value = expected_ev - net_debt
    expected_per_share = expected_equity_value / synthetic_bundle.info["sharesOutstanding"]

    assert result.equity_value == pytest.approx(expected_equity_value, rel=1e-6)
    assert result.intrinsic_value_per_share == pytest.approx(expected_per_share, rel=1e-6)

    expected_upside = (expected_per_share / synthetic_bundle.info["currentPrice"]) - 1
    assert result.upside == pytest.approx(expected_upside, rel=1e-6)


def test_invalid_wacc_le_terminal_growth_is_handled_gracefully(synthetic_bundle):
    """WACC must exceed terminal growth for Gordon Growth to be valid; the
    model must not crash or produce inf/NaN even if a user-supplied
    terminal growth rate is set at or above WACC."""
    wacc_result = calculate_wacc(synthetic_bundle, risk_free_rate=0.042, equity_risk_premium=0.045)
    # Deliberately set terminal growth above the computed WACC.
    assumptions = _single_year_assumptions(terminal_growth=wacc_result.wacc + 0.05)

    result = run_dcf(synthetic_bundle, assumptions, wacc_result)

    assert np.isfinite(result.terminal_value_gordon)
    assert np.isfinite(result.intrinsic_value_per_share)
    assert result.terminal_value_gordon > 0


def test_near_zero_wacc_terminal_growth_spread_is_floored(synthetic_bundle):
    """A WACC/terminal-growth spread that is small but positive (not
    literally invalid) must be floored to MIN_WACC_TERMINAL_GROWTH_SPREAD
    rather than passed through as-is to the Gordon Growth formula.

    This is the bug the Bull scenario (src/scenarios.py) could trigger in
    practice: it simultaneously lowers WACC and raises terminal growth, so
    a company with an already-thin base-case spread could see WACC and g
    end up just fractions of a percent apart, exploding the Gordon Growth
    denominator. Regression test for a real case found in manual testing
    (a Bull-case intrinsic value dozens of times the actual share price).
    """
    from src.dcf import MIN_WACC_TERMINAL_GROWTH_SPREAD

    wacc_result = calculate_wacc(synthetic_bundle, risk_free_rate=0.042, equity_risk_premium=0.045)
    # Terminal growth just 0.1 percentage points below WACC -- valid under
    # the literal wacc > g check, but numerically unstable without a floor.
    requested_g = wacc_result.wacc - 0.001
    assumptions = _single_year_assumptions(terminal_growth=requested_g)

    result = run_dcf(synthetic_bundle, assumptions, wacc_result)

    assert np.isfinite(result.terminal_value_gordon)
    assert np.isfinite(result.intrinsic_value_per_share)

    # The effective spread actually used must be the floor, not the
    # requested (much smaller) 0.001 spread -- verified by reconstructing
    # the terminal value from the floored inputs and checking it matches
    # exactly, rather than the value implied by the raw requested spread.
    fcff = result.projection["free_cash_flow"].iloc[0]
    floored_g = wacc_result.wacc - MIN_WACC_TERMINAL_GROWTH_SPREAD
    expected_tv_with_floor = (fcff * (1 + floored_g)) / (wacc_result.wacc - floored_g)
    unfloored_tv_at_requested_spread = (fcff * (1 + requested_g)) / (wacc_result.wacc - requested_g)

    assert result.terminal_value_gordon == pytest.approx(expected_tv_with_floor, rel=1e-6)
    # The floor must materially reduce the terminal value versus what the
    # raw (near-zero-spread) request would have produced.
    assert result.terminal_value_gordon < unfloored_tv_at_requested_spread / 10


def _capital_intensive_buildout_bundle() -> StockDataBundle:
    """A young, capital-intensive company mid build-out: CapEx running at
    several times revenue (modeled on IREN Limited, a data-center
    operator, found in manual testing), with operating margin improving
    from a heavy loss toward breakeven. Revenue: 60 -> 190 -> 500. CapEx:
    290 -> 480 -> 1370 (i.e. 480%, 250%, 274% of that year's revenue)."""
    years = [pd.Timestamp("2023-06-30"), pd.Timestamp("2024-06-30"), pd.Timestamp("2025-06-30")]
    income_stmt = pd.DataFrame({
        years[0]: {"Total Revenue": 60.0, "EBIT": -30.0, "Net Income": -30.0,
                   "Pretax Income": -30.0, "Tax Provision": 0.0},
        years[1]: {"Total Revenue": 190.0, "EBIT": -20.0, "Net Income": -20.0,
                   "Pretax Income": -20.0, "Tax Provision": 0.0},
        years[2]: {"Total Revenue": 500.0, "EBIT": 5.0, "Net Income": 5.0,
                   "Pretax Income": 5.0, "Tax Provision": 1.0},
    })
    cash_flow = pd.DataFrame({
        years[0]: {"Operating Cash Flow": -10.0, "Capital Expenditure": -290.0,
                   "Depreciation Amortization Depletion": 8.0},
        years[1]: {"Operating Cash Flow": 50.0, "Capital Expenditure": -480.0,
                   "Depreciation Amortization Depletion": 55.0},
        years[2]: {"Operating Cash Flow": 245.0, "Capital Expenditure": -1370.0,
                   "Depreciation Amortization Depletion": 180.0},
    })
    return StockDataBundle(
        ticker="TEST", info={"beta": 4.0, "currentPrice": 44.0, "sharesOutstanding": 300.0},
        income_stmt=income_stmt, balance_sheet=pd.DataFrame(), cash_flow=cash_flow,
        price_history=pd.DataFrame(),
    )


def test_capex_and_da_percentages_are_capped():
    """Regression test for a real bug report: a young, capital-intensive
    company mid build-out (CapEx running at 250-480% of revenue) must not
    have that ratio averaged and projected flat for 5 years -- doing so
    produced billions of dollars of projected negative FCF against
    roughly $1B of actual revenue in manual testing. The assumption should
    be capped at MAX_CAPEX_OR_DA_PCT_OF_REVENUE, with the original
    (uncapped) ratio preserved in the assumption's disclosure note.
    """
    bundle = _capital_intensive_buildout_bundle()
    assumptions = build_baseline_assumptions(bundle)

    assert assumptions.capex_pct_revenue.value == pytest.approx(MAX_CAPEX_OR_DA_PCT_OF_REVENUE)
    assert "capped" in assumptions.capex_pct_revenue.note.lower()
    # The uncapped historical ratio ((290/60 + 480/190 + 1370/500) / 3 ~= 337%)
    # should still be visible in the note for transparency.
    uncapped_ratio = (290 / 60 + 480 / 190 + 1370 / 500) / 3
    assert f"{uncapped_ratio:.0%}" in assumptions.capex_pct_revenue.note


def test_operating_margin_is_recency_weighted_not_a_flat_average():
    """Regression test: a company moving from a heavy-loss build-out phase
    toward profitability (EBIT margin -50% -> -10.5% -> +1%) should have
    its forecast operating margin anchored CLOSER to its current run-rate
    than a flat historical average would produce, since the earliest
    years are the least representative of where the business is today.
    """
    bundle = _capital_intensive_buildout_bundle()
    assumptions = build_baseline_assumptions(bundle)

    margins = [-30 / 60, -20 / 190, 5 / 500]  # -0.50, -0.105, 0.01
    flat_average = sum(margins) / len(margins)
    forecast_margin = assumptions.operating_margin_path[0]

    assert forecast_margin > flat_average  # weighted toward the improving, more recent years
    assert forecast_margin < margins[-1]   # but not simply equal to the single latest year either
