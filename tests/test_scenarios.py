"""Unit tests for src/scenarios.py: Bull / Base / Bear scenario modeling."""

import numpy as np
import pytest

from src.dcf import build_baseline_assumptions, calculate_wacc
from src.scenarios import annualized_potential_upside, run_scenarios, scenario_dispersion


def test_bull_base_bear_ordering(synthetic_bundle):
    """Bull-case assumptions (higher growth/margin, lower WACC) must produce
    a strictly higher intrinsic value than Base, which must exceed Bear
    (lower growth/margin, higher WACC) -- this is the core internal-
    consistency property of the scenario framework."""
    assumptions = build_baseline_assumptions(synthetic_bundle)
    wacc = calculate_wacc(synthetic_bundle)

    results = run_scenarios(synthetic_bundle, assumptions, wacc)

    bear_value = results["bear"].dcf.intrinsic_value_per_share
    base_value = results["base"].dcf.intrinsic_value_per_share
    bull_value = results["bull"].dcf.intrinsic_value_per_share

    assert bear_value < base_value < bull_value


def test_base_case_matches_unadjusted_dcf(synthetic_bundle):
    """The Base scenario applies zero adjustment, so it must exactly
    reproduce a plain DCF run on the same baseline assumptions."""
    from src.dcf import run_dcf

    assumptions = build_baseline_assumptions(synthetic_bundle)
    wacc = calculate_wacc(synthetic_bundle)

    results = run_scenarios(synthetic_bundle, assumptions, wacc)
    direct = run_dcf(synthetic_bundle, assumptions, wacc)

    assert results["base"].dcf.intrinsic_value_per_share == pytest.approx(
        direct.intrinsic_value_per_share, rel=1e-9
    )


def test_scenario_dispersion_is_positive_and_finite(synthetic_bundle):
    assumptions = build_baseline_assumptions(synthetic_bundle)
    wacc = calculate_wacc(synthetic_bundle)
    results = run_scenarios(synthetic_bundle, assumptions, wacc)

    dispersion = scenario_dispersion(results)
    assert np.isfinite(dispersion)
    assert dispersion > 0


# ---------------------------------------------------------------------------
# Potential Upside (1-year): non-negative, optimistic-case, attributable.
# ---------------------------------------------------------------------------
def _fake_results(bull_value, price):
    class _D:
        intrinsic_value_per_share = bull_value
        current_price = price
    class _R:
        dcf = _D()
    return {"bull": _R()}


def test_potential_upside_is_never_negative():
    """The metric is explicitly the optimistic case and is displayed without a
    sign, so a company whose bull case sits below its price reports 0.0% --
    'no upside found' -- rather than a negative number."""
    pu = annualized_potential_upside(_fake_results(50.0, 200.0), forecast_years=5,
                                     analyst_target_price=150.0, current_price=200.0)
    assert pu.value == 0.0
    assert pu.source == "none"
    # Both underlying candidates are retained for display even when the
    # headline figure is floored.
    assert pu.analyst_upside < 0
    assert pu.bull_case_upside < 0


def test_potential_upside_prefers_whichever_case_is_stronger():
    """Two independent upside cases; the metric reports the higher and says
    which one it used, so the number is always attributable."""
    strong_analyst = annualized_potential_upside(
        _fake_results(110.0, 100.0), forecast_years=5,
        analyst_target_price=140.0, current_price=100.0)
    assert strong_analyst.source == "analyst"
    assert strong_analyst.value == pytest.approx(0.40)

    strong_bull = annualized_potential_upside(
        _fake_results(400.0, 100.0), forecast_years=5,
        analyst_target_price=105.0, current_price=100.0)
    assert strong_bull.source == "bull_case"
    assert strong_bull.value == pytest.approx(4.0 ** (1 / 5) - 1)


def test_potential_upside_annualizes_rather_than_quoting_the_whole_thesis():
    """A multi-year DCF gap quoted as a one-year move would overstate it
    several times over: a 4x bull case over 5 years is ~32%/yr, not 300%."""
    pu = annualized_potential_upside(_fake_results(400.0, 100.0), forecast_years=5,
                                     analyst_target_price=None, current_price=None)
    assert pu.value == pytest.approx(4.0 ** (1 / 5) - 1)
    assert pu.value < 0.35


def test_potential_upside_falls_back_to_bull_case_without_analyst_coverage():
    pu = annualized_potential_upside(_fake_results(200.0, 100.0), forecast_years=5,
                                     analyst_target_price=None, current_price=100.0)
    assert pu.source == "bull_case"
    assert np.isnan(pu.analyst_upside)


def test_potential_upside_reports_unavailable_rather_than_zero_without_data():
    """'No data' must stay distinguishable from 'no upside' -- one is a gap in
    the feed, the other is a finding."""
    pu = annualized_potential_upside({}, forecast_years=5,
                                     analyst_target_price=None, current_price=None)
    assert np.isnan(pu.value)
    assert pu.source == "unavailable"
