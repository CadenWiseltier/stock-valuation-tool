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


@dataclass
class PotentialUpside:
    """A one-year potential-upside figure plus the evidence it came from."""

    #: Non-negative annual rate. NaN when nothing could be computed.
    value: float
    #: "analyst" | "bull_case" | "none" | "unavailable" -- shown in the UI so
    #: the number is never an unattributed assertion.
    source: str
    #: The two candidates, for display. Either may be NaN.
    analyst_upside: float = float("nan")
    bull_case_upside: float = float("nan")


def annualized_potential_upside(
    results: dict[str, ScenarioResult],
    forecast_years: int,
    analyst_target_price: float | None = None,
    current_price: float | None = None,
) -> PotentialUpside:
    """Potential upside over ONE year, as a non-negative rate.

    This answers "if things go well, how much could this gain in a year?" --
    a deliberately different question from `DCFResult.upside`, which is the
    total gap between the BASE-case intrinsic value and today's price and is
    frequently, legitimately negative.

    TWO INDEPENDENT UPSIDE CASES, WHICHEVER IS STRONGER
    ---------------------------------------------------
    1. Analyst consensus. A published 12-month mean price target is, by
       convention, exactly a one-year figure, and it reflects forward
       information this tool has no other way to see -- guidance, order
       books, product cycles. Where it exists it is usually backed by
       20-50 analysts.
    2. The bull-case DCF, annualized over the forecast horizon. This is the
       fallback for companies with thin or no analyst coverage, and it keeps
       the metric working entirely offline from this project's own model.

    The larger of the two is reported, because the metric is explicitly the
    OPTIMISTIC case rather than a central estimate. That choice is only
    honest while the label says so, so the UI names it "Potential Upside" and
    the help text states plainly that it is not an expected or predicted
    return. `source` records which input won, so the figure is always
    attributable.

    Taking the maximum also corrects a real distortion. This project's DCF is
    deliberately conservative -- its median base-case upside across 49 real
    companies was -54% -- so relying on it alone reported 0.0% potential
    upside for Microsoft, Apple, NVIDIA and JPMorgan simultaneously. That is
    a far stronger claim than the model can support, and it is contradicted
    by dozens of analysts covering each of those companies.

    WHY IT IS FLOORED AT ZERO
    -------------------------
    When neither case clears today's price, the honest reading is "this
    analysis sees no upside", which is what 0.0% says. Nothing is concealed:
    the base-case intrinsic value is displayed immediately beside it, the
    bear case drives its own subscore in the Investment Score, and the full
    scenario range appears in the scenario table.

    Returns value=NaN with source="unavailable" when neither input can be
    computed, so "no data" stays distinguishable from "no upside".
    """
    analyst_upside = np.nan
    if (analyst_target_price is not None and current_price is not None
            and not np.isnan(analyst_target_price) and not np.isnan(current_price)
            and current_price > 0 and analyst_target_price > 0):
        analyst_upside = analyst_target_price / current_price - 1.0

    bull_upside = np.nan
    bull = results.get("bull") if results else None
    if bull is not None and forecast_years and forecast_years > 0:
        value = bull.dcf.intrinsic_value_per_share
        price = bull.dcf.current_price
        if (value is not None and price is not None and not np.isnan(value)
                and not np.isnan(price) and price > 0 and value > 0):
            bull_upside = (value / price) ** (1.0 / forecast_years) - 1.0

    candidates = [(analyst_upside, "analyst"), (bull_upside, "bull_case")]
    usable = [(v, name) for v, name in candidates if not np.isnan(v)]
    if not usable:
        return PotentialUpside(value=np.nan, source="unavailable",
                               analyst_upside=analyst_upside, bull_case_upside=bull_upside)

    best_value, best_source = max(usable, key=lambda pair: pair[0])
    if best_value <= 0:
        return PotentialUpside(value=0.0, source="none",
                               analyst_upside=analyst_upside, bull_case_upside=bull_upside)
    return PotentialUpside(value=best_value, source=best_source,
                           analyst_upside=analyst_upside, bull_case_upside=bull_upside)
