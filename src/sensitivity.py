"""
DCF sensitivity analysis.

A single DCF output is a point estimate that depends heavily on two of the
hardest-to-forecast inputs: WACC and the terminal growth rate. This module
recomputes intrinsic value per share across a grid of each, so the user can
see how much the valuation actually depends on those assumptions rather than
anchoring on one number.

A second grid does the same for Revenue Growth x Operating Margin, the two
assumptions that drive the explicit forecast period.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.data import StockDataBundle
from src.dcf import DCFAssumptions, WACCResult, run_dcf


def wacc_vs_terminal_growth_grid(
    bundle: StockDataBundle,
    base_assumptions: DCFAssumptions,
    base_wacc: WACCResult,
    wacc_range: list[float] | None = None,
    growth_range: list[float] | None = None,
) -> pd.DataFrame:
    """Grid of intrinsic value per share across WACC (rows) x terminal
    growth rate (columns). Cells where WACC <= terminal growth are set to
    NaN because the Gordon Growth terminal-value formula is undefined
    (or economically nonsensical) in that region.
    """
    if wacc_range is None:
        center = base_wacc.wacc
        wacc_range = [center + d for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
    if growth_range is None:
        center_g = base_assumptions.terminal_growth.value
        growth_range = [center_g + d for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]

    data = {}
    for g in growth_range:
        col = []
        for w in wacc_range:
            if w <= g:
                col.append(np.nan)
                continue
            assumptions = copy.deepcopy(base_assumptions)
            assumptions.terminal_growth.value = g
            wacc_result = WACCResult(
                cost_of_equity=base_wacc.cost_of_equity,
                after_tax_cost_of_debt=base_wacc.after_tax_cost_of_debt,
                pretax_cost_of_debt=base_wacc.pretax_cost_of_debt,
                equity_value_weight=base_wacc.equity_value_weight,
                debt_value_weight=base_wacc.debt_value_weight,
                market_cap=base_wacc.market_cap,
                total_debt=base_wacc.total_debt,
                wacc=w,
            )
            result = run_dcf(bundle, assumptions, wacc_result)
            col.append(result.intrinsic_value_per_share)
        data[f"{g:.1%}"] = col

    df = pd.DataFrame(data, index=[f"{w:.1%}" for w in wacc_range])
    df.index.name = "WACC \\ Terminal Growth"
    return df


def revenue_growth_vs_margin_grid(
    bundle: StockDataBundle,
    base_assumptions: DCFAssumptions,
    base_wacc: WACCResult,
    growth_deltas: list[float] | None = None,
    margin_deltas: list[float] | None = None,
) -> pd.DataFrame:
    """Grid of intrinsic value per share across a shift in the Year-1
    revenue growth rate (rows) x a shift in operating margin (columns),
    holding WACC and terminal growth at their base-case values.
    """
    if growth_deltas is None:
        growth_deltas = [-0.04, -0.02, 0.0, 0.02, 0.04]
    if margin_deltas is None:
        margin_deltas = [-0.03, -0.015, 0.0, 0.015, 0.03]

    data = {}
    for md in margin_deltas:
        col = []
        for gd in growth_deltas:
            assumptions = copy.deepcopy(base_assumptions)
            assumptions.revenue_growth_path = [g + gd for g in assumptions.revenue_growth_path]
            assumptions.operating_margin_path = [
                max(m + md, 0.0) for m in assumptions.operating_margin_path
            ]
            result = run_dcf(bundle, assumptions, base_wacc)
            col.append(result.intrinsic_value_per_share)
        base_margin = base_assumptions.operating_margin_path[0]
        data[f"{base_margin + md:.1%}"] = col

    base_growth = base_assumptions.revenue_growth_path[0]
    df = pd.DataFrame(data, index=[f"{base_growth + gd:.1%}" for gd in growth_deltas])
    df.index.name = "Yr-1 Revenue Growth \\ Operating Margin"
    return df
