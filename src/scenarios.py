"""
Bull / Base / Bear scenario valuation.

Re-runs the same DCF engine (src/dcf.py) three times with systematically
adjusted assumptions, rather than three independently hand-picked sets of
numbers -- this keeps the scenarios internally consistent and makes the
adjustment logic auditable (see `SCENARIO_ADJUSTMENTS` below).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from src.data import StockDataBundle
from src.dcf import DCFAssumptions, DCFResult, WACCResult, calculate_wacc, run_dcf

# ---------------------------------------------------------------------------
# Scenario adjustment rules.
#
# Each scenario shifts the BASE-case assumptions by a fixed, documented
# amount. These are model assumptions, not statistically derived, and are
# intentionally symmetric (bear and bull move by the same magnitude in
# opposite directions) so neither case is arbitrarily favored.
# ---------------------------------------------------------------------------
SCENARIO_ADJUSTMENTS = {
    "bear": {
        "revenue_growth_delta": -0.03,   # -3 percentage points per forecast year
        "operating_margin_delta": -0.02,  # -2 percentage points
        "wacc_delta": +0.010,             # +100 bps (higher perceived risk)
        "terminal_growth_delta": -0.005,  # -50 bps
    },
    "base": {
        "revenue_growth_delta": 0.0,
        "operating_margin_delta": 0.0,
        "wacc_delta": 0.0,
        "terminal_growth_delta": 0.0,
    },
    "bull": {
        "revenue_growth_delta": +0.03,
        "operating_margin_delta": +0.02,
        "wacc_delta": -0.005,             # -50 bps (lower perceived risk)
        "terminal_growth_delta": +0.005,
    },
    # A fourth, more extreme tier used ONLY by the Speculative Score
    # (src/speculative.py) -- the Buy Rating's Scenario-Weighted category
    # deliberately still uses only bear/base/bull (see SCENARIO_PROBABILITIES
    # there), so adding this key does not change any existing behavior.
    # Extends the same linear bull-case deltas one step further, rather
    # than an arbitrarily chosen "moonshot" number, so it stays internally
    # consistent with how the other three tiers were built.
    "extreme_bull": {
        "revenue_growth_delta": +0.06,
        "operating_margin_delta": +0.04,
        "wacc_delta": -0.010,
        "terminal_growth_delta": +0.010,
    },
}


@dataclass
class ScenarioResult:
    name: str
    dcf: DCFResult


def _shift_assumptions(base: DCFAssumptions, adj: dict) -> DCFAssumptions:
    shifted = copy.deepcopy(base)
    shifted.revenue_growth_path = [
        g + adj["revenue_growth_delta"] for g in shifted.revenue_growth_path
    ]
    shifted.operating_margin_path = [
        max(m + adj["operating_margin_delta"], 0.0) for m in shifted.operating_margin_path
    ]
    shifted.terminal_growth.value = max(shifted.terminal_growth.value + adj["terminal_growth_delta"], 0.0)
    return shifted


def run_scenarios(
    bundle: StockDataBundle,
    base_assumptions: DCFAssumptions,
    base_wacc: WACCResult,
) -> dict[str, ScenarioResult]:
    """Run the Bear / Base / Bull DCF scenarios and return all three results."""
    results = {}
    for name, adj in SCENARIO_ADJUSTMENTS.items():
        scenario_assumptions = _shift_assumptions(base_assumptions, adj)
        scenario_wacc = WACCResult(
            cost_of_equity=base_wacc.cost_of_equity,
            after_tax_cost_of_debt=base_wacc.after_tax_cost_of_debt,
            pretax_cost_of_debt=base_wacc.pretax_cost_of_debt,
            equity_value_weight=base_wacc.equity_value_weight,
            debt_value_weight=base_wacc.debt_value_weight,
            market_cap=base_wacc.market_cap,
            total_debt=base_wacc.total_debt,
            wacc=base_wacc.wacc + adj["wacc_delta"],
        )
        dcf_result = run_dcf(bundle, scenario_assumptions, scenario_wacc)
        results[name] = ScenarioResult(name=name, dcf=dcf_result)
    return results


def scenario_dispersion(results: dict[str, ScenarioResult]) -> float:
    """Spread between bull and bear intrinsic values, as a fraction of the
    base-case value. A narrower spread implies the valuation is less
    sensitive to the direction of the underlying business assumptions."""
    bear_v = results["bear"].dcf.intrinsic_value_per_share
    bull_v = results["bull"].dcf.intrinsic_value_per_share
    base_v = results["base"].dcf.intrinsic_value_per_share
    if any(np.isnan(x) or x == 0 for x in [bear_v, bull_v, base_v]):
        return np.nan
    return (bull_v - bear_v) / abs(base_v)
