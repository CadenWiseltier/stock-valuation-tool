"""Unit tests for src/scenarios.py: Bull / Base / Bear scenario modeling."""

import numpy as np
import pytest

from src.dcf import build_baseline_assumptions, calculate_wacc
from src.scenarios import run_scenarios, scenario_dispersion


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
