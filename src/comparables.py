"""
Comparable-company ("comps") valuation.

Cross-checks the DCF against how the market currently prices similar
businesses. Peers are selected from the target's GICS-style sector (see
`src.data.get_peer_tickers`) so the comparison is economically meaningful --
never an arbitrary basket of unrelated companies.

For each valuation multiple (P/E, EV/EBITDA, EV/Revenue, Price/Sales) the
module computes the peer mean/median and applies it to the target company's
own fundamentals (EPS, EBITDA, revenue) to produce an "implied" share price
under that multiple -- the standard comps methodology used in equity
research.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import StockDataBundle, fetch_peer_snapshot, get_peer_tickers, infer_peer_sector, row
from src.data import (
    EBITDA as _EBITDA, REVENUE as _REVENUE, NET_INCOME as _NET_INCOME,
    TOTAL_DEBT as _TOTAL_DEBT, CASH as _CASH,
)


def _to_float_or_nan(x) -> float:
    """Coerce a possibly-None/non-numeric API value to NaN rather than 0.

    `bundle.info.get("totalDebt")` returning `None` means "the data
    provider did not report this field" -- treating that the same as a
    reported value of exactly 0 (`x or 0.0`) would silently assume a
    company has no debt whenever the field happens to be missing, which
    is a very different claim from "this company's debt is unknown."
    """
    if x is None:
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


@dataclass
class ComparablesResult:
    peer_table: pd.DataFrame              # one row per peer + multiples
    mean_multiples: pd.Series
    median_multiples: pd.Series
    implied_values: dict                  # {"P/E": value_per_share, ...}
    comps_valuation_low: float
    comps_valuation_high: float
    peer_tickers: list[str]
    sector: str | None          # sector as reported by the data provider
    peer_sector: str | None     # sector actually used for peer selection (see infer_peer_sector)


def _latest(series):
    if series is None:
        return np.nan
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def build_comparables(bundle: StockDataBundle, max_peers: int = 6) -> ComparablesResult:
    """Assemble the peer multiple table and implied valuations for `bundle`.

    Why these peers: companies are drawn from the same sector classification
    reported by the data provider, which groups businesses with broadly
    similar demand drivers, margin structures, and capital intensity --
    a reasonable, transparent proxy for "comparable" in the absence of a
    paid, hand-curated peer-screening service.
    """
    sector = bundle.info.get("sector")
    peer_tickers = get_peer_tickers(bundle, max_peers=max_peers)

    rows = []
    for pt in peer_tickers:
        try:
            snap = fetch_peer_snapshot(pt)
            rows.append(snap)
        except Exception:
            continue

    peer_table = pd.DataFrame(rows)
    if not peer_table.empty:
        peer_table = peer_table.rename(columns={
            "ticker": "Ticker", "name": "Company",
            "trailing_pe": "P/E", "ev_to_ebitda": "EV/EBITDA",
            "ev_to_revenue": "EV/Revenue", "price_to_sales": "P/S",
        })[["Ticker", "Company", "P/E", "EV/EBITDA", "EV/Revenue", "P/S"]]

        # Discard implausible/negative multiples (e.g. a peer with negative
        # earnings produces a meaningless negative P/E) before averaging.
        numeric_cols = ["P/E", "EV/EBITDA", "EV/Revenue", "P/S"]
        for c in numeric_cols:
            peer_table[c] = pd.to_numeric(peer_table[c], errors="coerce")
            peer_table.loc[peer_table[c] <= 0, c] = np.nan

        mean_multiples = peer_table[numeric_cols].mean()
        median_multiples = peer_table[numeric_cols].median()
    else:
        mean_multiples = pd.Series(dtype=float)
        median_multiples = pd.Series(dtype=float)

    # --- Target company fundamentals needed to apply the multiples ---
    net_income = _latest(row(bundle.income_stmt, *_NET_INCOME))
    ebitda = _latest(row(bundle.income_stmt, *_EBITDA))
    revenue = _latest(row(bundle.income_stmt, *_REVENUE))
    shares = bundle.info.get("sharesOutstanding") or np.nan

    # Prefer the lightweight `info` fields (Step 1); fall back to the full
    # balance sheet (Step 2/3: an alternate data source) before giving up.
    # Missing net debt must propagate as NaN, not silently become 0 --
    # assuming zero debt when it is actually unknown would OVERSTATE the
    # comps-implied enterprise-to-equity bridge for any leveraged company.
    total_debt = _to_float_or_nan(bundle.info.get("totalDebt"))
    if np.isnan(total_debt):
        total_debt = _latest(row(bundle.balance_sheet, *_TOTAL_DEBT))
    cash = _to_float_or_nan(bundle.info.get("totalCash"))
    if np.isnan(cash):
        cash = _latest(row(bundle.balance_sheet, *_CASH))
    net_debt = (total_debt - cash) if (not np.isnan(total_debt) and not np.isnan(cash)) else np.nan

    implied_values: dict[str, float] = {}

    if shares and not np.isnan(shares) and shares > 0:
        # P/E -> implied equity value per share directly.
        if not np.isnan(net_income) and "P/E" in median_multiples and pd.notna(median_multiples["P/E"]):
            eps = net_income / shares
            implied_values["P/E"] = eps * median_multiples["P/E"]

        # EV/EBITDA -> implied enterprise value, bridge to equity, then per share.
        if not np.isnan(ebitda) and "EV/EBITDA" in median_multiples and pd.notna(median_multiples["EV/EBITDA"]):
            implied_ev = ebitda * median_multiples["EV/EBITDA"]
            implied_values["EV/EBITDA"] = (implied_ev - net_debt) / shares

        # EV/Revenue -> implied enterprise value, bridge to equity, then per share.
        if not np.isnan(revenue) and "EV/Revenue" in median_multiples and pd.notna(median_multiples["EV/Revenue"]):
            implied_ev = revenue * median_multiples["EV/Revenue"]
            implied_values["EV/Revenue"] = (implied_ev - net_debt) / shares

    valid_values = [v for v in implied_values.values() if v is not None and not np.isnan(v) and v > 0]
    comps_low = float(min(valid_values)) if valid_values else np.nan
    comps_high = float(max(valid_values)) if valid_values else np.nan

    return ComparablesResult(
        peer_table=peer_table,
        mean_multiples=mean_multiples,
        median_multiples=median_multiples,
        implied_values=implied_values,
        comps_valuation_low=comps_low,
        comps_valuation_high=comps_high,
        peer_tickers=peer_tickers,
        sector=sector,
        peer_sector=infer_peer_sector(bundle.info),
    )
