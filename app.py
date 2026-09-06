"""
Stock Valuation & Investment Analysis Tool - Streamlit entry point.

Design philosophy: "simple on the outside, sophisticated on the inside."
The user only ever has to type a ticker and click Analyze. Everything else
(financial-statement analysis, DCF valuation, WACC, comparable-company
analysis, scenario modeling, sensitivity analysis, the 0-1000 Fundamental
Investment Score, and the 0-1000 Speculation Score) runs automatically.
Power users can expand "Advanced Assumptions" to inspect or override the
model's inputs, but nothing in the default path requires it.

Two separate 0-1000 scores are shown, answering two different questions
(see docs/methodology.md section 9 for the full rationale):
    Fundamental Investment Score (src/scoring.py) -- "Is this stock
        attractively priced enough to buy today, given its fundamentals,
        expected return, valuation, and risk?"
    Speculation Score (src/speculative.py) -- "How much potential does this
        stock have to EXPLODE in value if its bullish thesis, catalysts, or
        emerging-industry opportunity plays out, and how credible is that
        scenario?" It is forward-looking and opportunity-focused, driven by
        the industry knowledge base in src/themes.py, and is INDEPENDENT of
        the Fundamental Score -- a Fundamental 350 / Speculation 900 pairing
        is not a contradiction.

This file owns ONLY presentation/layout. All financial logic lives in
src/*.py so it can be unit tested independently of Streamlit.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src.comparables import ComparablesResult, build_comparables
from src.data import (
    DataProviderError, LivePrice, StockDataBundle, TickerNotFoundError,
    apply_live_price, fetch_live_price, fetch_stock_data, infer_peer_sector, row,
)
from src.data import EBITDA as _EBITDA, REVENUE as _REVENUE
from src.dcf import (
    DCFAssumptions, DCFResult, WACCResult, build_baseline_assumptions,
    calculate_wacc, run_dcf, DEFAULT_RISK_FREE_RATE, DEFAULT_EQUITY_RISK_PREMIUM,
    DEFAULT_TERMINAL_GROWTH, FORECAST_YEARS,
)
from src.financial_analysis import (
    balance_sheet_summary, credit_metrics, diluted_shares_growth_rate,
    earnings_to_cash_conversion, fcf_summary, generate_trend_insights,
    historical_pe_series, income_statement_summary, is_financial_services_company,
    roe_series, roic_series,
)
from src.scenarios import ScenarioResult, run_scenarios, scenario_dispersion
from src.scoring import InvestmentScoreResult, compute_investment_score, price_sensitivity_table
from src.sensitivity import revenue_growth_vs_margin_grid, wacc_vs_terminal_growth_grid
from src.speculative import SpeculativeScoreResult, compute_speculative_score

st.set_page_config(page_title="Stock Valuation & Investment Analysis", layout="wide", page_icon="📊")


# ---------------------------------------------------------------------------
# Cached data / computation layer.
#
# `fetch_stock_data` hits the network, so it is cached for a limited time
# (financial statements update quarterly at most; price data is not
# real-time in this app -- see the "data as of" notice rendered on every
# results page). `run_full_analysis` is cached separately, keyed on every
# assumption that can change the output, so the whole pipeline (DCF, comps,
# scenarios, sensitivity, score) only recomputes when the ticker or an
# assumption actually changes.
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_stock_data(ticker: str) -> StockDataBundle:
    return fetch_stock_data(ticker)


# The quote gets its own, far shorter cache than the statement bundle above.
# Splitting them is the entire point: statements are quarterly and expensive
# to fetch, while the price moves continuously and drives roughly 65% of the
# Fundamental Investment Score, so an hour-old quote silently shifts the
# RATING. 60 seconds keeps the rating honest without hammering either
# provider -- and because it is keyed only on the ticker, every visitor to a
# public deployment shares one request per minute rather than issuing their
# own.
@st.cache_data(ttl=60, show_spinner=False)
def cached_fetch_live_price(ticker: str) -> Optional[LivePrice]:
    return fetch_live_price(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_build_comparables(_bundle: StockDataBundle, ticker: str) -> ComparablesResult:
    # `ticker` is passed only so Streamlit's cache key reflects the company
    # being analyzed; `_bundle` is excluded from hashing (leading
    # underscore) since StockDataBundle holds unhashable DataFrames.
    return build_comparables(_bundle)


@dataclasses.dataclass
class AnalysisBundle:
    bundle: StockDataBundle
    income_df: pd.DataFrame
    fcf_df: pd.DataFrame
    balance_df: pd.DataFrame
    credit: dict
    roe: pd.Series
    roic: pd.Series
    earnings_conversion: pd.Series
    insights: list[str]
    assumptions: DCFAssumptions
    wacc: WACCResult
    dcf: DCFResult
    comps: ComparablesResult
    scenarios: dict[str, ScenarioResult]
    dispersion: float
    wacc_grid: pd.DataFrame
    growth_margin_grid: pd.DataFrame
    score: InvestmentScoreResult
    historical_pe: pd.Series
    dilution_rate: float
    price_sensitivity: pd.DataFrame
    speculative: SpeculativeScoreResult
    live_price: Optional[LivePrice] = None


def run_full_analysis(
    ticker: str,
    forecast_years: int,
    risk_free_rate: float,
    equity_risk_premium: float,
    terminal_growth: float,
    beta_override: Optional[float],
    tax_rate_override: Optional[float],
    revenue_growth_override: Optional[float],
    operating_margin_override: Optional[float],
    capex_pct_override: Optional[float],
    da_pct_override: Optional[float],
    nwc_pct_override: Optional[float],
) -> AnalysisBundle:
    bundle = cached_fetch_stock_data(ticker)

    # Refresh the quote before ANY downstream calculation runs. The bundle
    # above may be up to an hour old; every price-dependent figure in the
    # report -- DCF upside, all four price-dependent score categories, WACC's
    # equity weight, the Speculation Score's upside multiple -- is derived
    # from the price and market cap this call overwrites, so it has to happen
    # here rather than being patched into the display layer afterwards.
    # A None result is not an error: apply_live_price leaves the cached
    # figures untouched and the UI says so.
    live_price = cached_fetch_live_price(ticker)
    apply_live_price(bundle, live_price)

    income_df = income_statement_summary(bundle)
    fcf_df = fcf_summary(bundle)
    balance_df = balance_sheet_summary(bundle)
    credit = credit_metrics(bundle)
    roe = roe_series(bundle)
    roic = roic_series(bundle)
    earnings_conversion = earnings_to_cash_conversion(bundle)
    insights = generate_trend_insights(income_df, fcf_df)

    assumptions = build_baseline_assumptions(
        bundle, forecast_years=forecast_years, terminal_growth=terminal_growth,
        risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium,
    )
    # Apply advanced-user overrides on top of the automatically generated
    # baseline, re-labeling each overridden field as an external assumption
    # so the UI is transparent about what the user changed.
    if tax_rate_override is not None:
        assumptions.tax_rate.value, assumptions.tax_rate.kind = tax_rate_override, "external_assumption"
    if capex_pct_override is not None:
        assumptions.capex_pct_revenue.value, assumptions.capex_pct_revenue.kind = capex_pct_override, "external_assumption"
    if da_pct_override is not None:
        assumptions.da_pct_revenue.value, assumptions.da_pct_revenue.kind = da_pct_override, "external_assumption"
    if nwc_pct_override is not None:
        assumptions.nwc_pct_revenue_change.value, assumptions.nwc_pct_revenue_change.kind = nwc_pct_override, "external_assumption"
    if revenue_growth_override is not None:
        assumptions.revenue_growth_path = list(
            np.linspace(revenue_growth_override, terminal_growth, forecast_years)
        )
    if operating_margin_override is not None:
        assumptions.operating_margin_path = [operating_margin_override] * forecast_years

    wacc = calculate_wacc(
        bundle, risk_free_rate=risk_free_rate, equity_risk_premium=equity_risk_premium,
        beta_override=beta_override, tax_rate=assumptions.tax_rate.value,
    )
    dcf_result = run_dcf(bundle, assumptions, wacc)

    comps = cached_build_comparables(bundle, ticker)

    scenario_results = run_scenarios(bundle, assumptions, wacc)
    dispersion = scenario_dispersion(scenario_results)

    wacc_grid = wacc_vs_terminal_growth_grid(bundle, assumptions, wacc)
    growth_margin_grid = revenue_growth_vs_margin_grid(bundle, assumptions, wacc)

    # Trailing-twelve-month fundamentals used to derive PRICE-DEPENDENT
    # multiples from scratch (src/scoring.py: _derive_price_dependent_multiples)
    # rather than trusting a data provider's own P/E/EV-EBITDA fields, which
    # keeps every price-dependent subscore internally consistent and is what
    # makes price_sensitivity_table() below possible.
    eps_ttm = income_df["diluted_eps"].dropna().iloc[-1] if not income_df.empty and not income_df["diluted_eps"].dropna().empty else np.nan
    ebitda_ttm = row(bundle.income_stmt, *_EBITDA)
    ebitda_ttm = ebitda_ttm.dropna().iloc[-1] if ebitda_ttm is not None and not ebitda_ttm.dropna().empty else np.nan
    revenue_ttm = row(bundle.income_stmt, *_REVENUE)
    revenue_ttm = revenue_ttm.dropna().iloc[-1] if revenue_ttm is not None and not revenue_ttm.dropna().empty else np.nan

    historical_pe = historical_pe_series(bundle)
    dilution_rate = diluted_shares_growth_rate(bundle)

    score = compute_investment_score(
        income_df=income_df, fcf_df=fcf_df, balance_df=balance_df, credit=credit,
        roe=roe, roic=roic, earnings_conversion=earnings_conversion,
        dcf_result=dcf_result, comps_result=comps, scenario_results=scenario_results,
        scenario_dispersion=dispersion, historical_pe=historical_pe, dilution_rate=dilution_rate,
        eps_ttm=eps_ttm, ebitda_ttm=ebitda_ttm, revenue_ttm=revenue_ttm,
        # Sector/industry select a scoring profile so the score stops asking
        # questions that do not apply to the business (gross margin on a
        # bank, industrial leverage limits on a REIT). `infer_peer_sector`
        # is used rather than the raw reported sector for the same reason it
        # is used for peer selection -- the provider's classification is
        # occasionally wrong in ways that would pick the wrong profile.
        sector=infer_peer_sector(bundle.info), industry=bundle.info.get("industry"),
    )

    business_quality_points = sum(
        c.points for c in score.categories if c.name in ("Growth", "Profitability", "Financial Health", "Cash Flow Quality")
    )
    risk_points = next(c.points for c in score.categories if c.name == "Risk")
    price_sensitivity = price_sensitivity_table(
        income_df=income_df, fcf_df=fcf_df, dcf_result=dcf_result, comps_result=comps,
        scenario_results=scenario_results, historical_pe=historical_pe, dilution_rate=dilution_rate,
        eps_ttm=eps_ttm, ebitda_ttm=ebitda_ttm, revenue_ttm=revenue_ttm,
        business_quality_points=business_quality_points, risk_points=risk_points,
        sector=infer_peer_sector(bundle.info), industry=bundle.info.get("industry"),
        roe=roe, roic=roic,
    )

    speculative = compute_speculative_score(
        bundle=bundle, income_df=income_df, fcf_df=fcf_df, credit=credit,
        dcf_result=dcf_result, scenario_results=scenario_results,
        dilution_rate=dilution_rate,
    )

    return AnalysisBundle(
        bundle=bundle, income_df=income_df, fcf_df=fcf_df, balance_df=balance_df,
        credit=credit, roe=roe, roic=roic, earnings_conversion=earnings_conversion,
        insights=insights, assumptions=assumptions, wacc=wacc, dcf=dcf_result,
        comps=comps, scenarios=scenario_results, dispersion=dispersion,
        wacc_grid=wacc_grid, growth_margin_grid=growth_margin_grid, score=score,
        historical_pe=historical_pe, dilution_rate=dilution_rate, price_sensitivity=price_sensitivity,
        speculative=speculative, live_price=live_price,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_money(x, suffix="") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"${x / 1e12:,.2f}T{suffix}"
    if abs_x >= 1e9:
        return f"${x / 1e9:,.2f}B{suffix}"
    if abs_x >= 1e6:
        return f"${x / 1e6:,.2f}M{suffix}"
    return f"${x:,.2f}{suffix}"


def fmt_pct(x, decimals=1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x, decimals=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{decimals}f}"


def fmt_ratio(x, decimals=2, suffix="x") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{decimals}f}{suffix}"


def fmt_subscore_value(sub) -> str:
    """Format a `SubScore`'s actual underlying value for display.

    CRITICAL: a subscore missing its data (`sub.missing=True`) is the ONLY
    case that renders as "N/A". A genuinely negative or zero value is
    never collapsed to "N/A" or blanked out -- it prints with its real
    sign, exactly as computed, since a score of e.g. 0/150 next to a
    hidden "-8.4%" would look identical to (and be indistinguishable from)
    a genuine data failure. See docs/methodology.md for why "missing" and
    "computed but very negative" are represented completely differently
    throughout this app's scoring pipeline.
    """
    if sub.missing or sub.value is None or (isinstance(sub.value, float) and np.isnan(sub.value)):
        return "N/A"
    if sub.unit == "pct":
        return fmt_pct(sub.value)
    if sub.unit == "ratio":
        return fmt_ratio(sub.value)
    if sub.unit == "usd":
        return fmt_money(sub.value)
    return fmt_num(sub.value, decimals=2)


def md_escape(text: str) -> str:
    """Escape dollar signs before passing prose to `st.markdown`.

    Streamlit's markdown renderer treats a pair of unescaped `$` as LaTeX
    math delimiters. A sentence containing two money figures (e.g. "a market
    of $90.0B ... revenue of $130.0M") therefore renders everything between
    them as a math span -- found while visually verifying the Speculation
    Score explanation, where it silently turned half a paragraph into
    monospace. Escaping is done here, at the render site, so the modules
    that build these strings stay presentation-agnostic.
    """
    return text.replace("$", r"\$")


ASSUMPTION_LABELS = {
    "historical": "📊 Historical Data",
    "model_assumption": "🧮 Model Assumption",
    "external_assumption": "🌐 External Assumption",
}


# ---------------------------------------------------------------------------
# UI: header
# ---------------------------------------------------------------------------
st.title("Stock Valuation & Investment Analysis")
st.caption(
    "Enter a ticker to get a full DCF valuation, comparable-company analysis, "
    "scenario modeling, and a 0-1000 quantitative Fundamental Investment Score, plus a separate "
    "forward-looking Speculation Score. Educational tool -- not financial advice."
)

col1, col2 = st.columns([3, 1])
with col1:
    ticker_input = st.text_input("Enter a stock ticker", value="AMZN", label_visibility="visible",
                                  placeholder="e.g. AMZN").strip().upper()
with col2:
    st.write("")
    st.write("")
    analyze_clicked = st.button("Analyze Stock", type="primary", use_container_width=True)

with st.expander("⚙️ Advanced Assumptions (optional -- normal users can skip this)"):
    st.caption(
        "These fields let you override the automatically generated model assumptions. "
        "Leave any field at its default to keep the automatic, history-derived value."
    )
    a1, a2, a3 = st.columns(3)
    with a1:
        forecast_years = st.slider("Forecast period (years)", min_value=3, max_value=10, value=FORECAST_YEARS)
        risk_free_rate_in = st.number_input("Risk-free rate", value=DEFAULT_RISK_FREE_RATE, format="%.4f", step=0.001)
        equity_risk_premium_in = st.number_input("Equity risk premium", value=DEFAULT_EQUITY_RISK_PREMIUM,
                                                  format="%.4f", step=0.001)
    with a2:
        terminal_growth_in = st.number_input("Terminal growth rate", value=DEFAULT_TERMINAL_GROWTH,
                                              format="%.4f", step=0.001)
        beta_in = st.number_input("Beta override (0 = use data provider value)", value=0.0, format="%.2f", step=0.05)
        tax_rate_in = st.number_input("Tax rate override (0 = use historical average)", value=0.0,
                                       format="%.4f", step=0.005)
    with a3:
        revenue_growth_in = st.number_input("Year-1 revenue growth override (0 = use automatic estimate)",
                                             value=0.0, format="%.4f", step=0.005)
        operating_margin_in = st.number_input("Operating margin override (0 = use automatic estimate)",
                                               value=0.0, format="%.4f", step=0.005)
        capex_pct_in = st.number_input("CapEx % of revenue override (0 = use automatic estimate)",
                                        value=0.0, format="%.4f", step=0.005)
    da_pct_in = st.number_input("D&A % of revenue override (0 = use automatic estimate)",
                                 value=0.0, format="%.4f", step=0.005)
    nwc_pct_in = st.number_input("Incremental NWC % of revenue-change override (0 = use default 5%)",
                                  value=0.0, format="%.4f", step=0.005)

if analyze_clicked:
    st.session_state["analyzed_ticker"] = ticker_input
    st.session_state["overrides"] = dict(
        forecast_years=forecast_years,
        risk_free_rate=risk_free_rate_in,
        equity_risk_premium=equity_risk_premium_in,
        terminal_growth=terminal_growth_in,
        beta_override=beta_in if beta_in else None,
        tax_rate_override=tax_rate_in if tax_rate_in else None,
        revenue_growth_override=revenue_growth_in if revenue_growth_in else None,
        operating_margin_override=operating_margin_in if operating_margin_in else None,
        capex_pct_override=capex_pct_in if capex_pct_in else None,
        da_pct_override=da_pct_in if da_pct_in else None,
        nwc_pct_override=nwc_pct_in if nwc_pct_in else None,
    )

if not st.session_state.get("analyzed_ticker"):
    st.info("Enter a ticker (e.g. AMZN, AAPL, MSFT) and click **Analyze Stock** to generate a full report.")
    st.stop()

ticker = st.session_state["analyzed_ticker"]
ov = st.session_state["overrides"]

# ---------------------------------------------------------------------------
# Run the analysis (with graceful error handling for bad tickers / API issues)
# ---------------------------------------------------------------------------
try:
    with st.spinner(f"Retrieving data and running full valuation model for {ticker}..."):
        analysis = run_full_analysis(
            ticker,
            forecast_years=ov["forecast_years"],
            risk_free_rate=ov["risk_free_rate"],
            equity_risk_premium=ov["equity_risk_premium"],
            terminal_growth=ov["terminal_growth"],
            beta_override=ov["beta_override"],
            tax_rate_override=ov["tax_rate_override"],
            revenue_growth_override=ov["revenue_growth_override"],
            operating_margin_override=ov["operating_margin_override"],
            capex_pct_override=ov["capex_pct_override"],
            da_pct_override=ov["da_pct_override"],
            nwc_pct_override=ov["nwc_pct_override"],
        )
except TickerNotFoundError as e:
    st.error(str(e))
    st.stop()
except DataProviderError as e:
    # A provider outage or rate limit is not the visitor's fault and is not a
    # bug they can do anything about, so it gets a plain, actionable message
    # rather than an error dump.
    st.warning(f"**{e}**")
    st.caption(
        "This app uses a free, unofficial market-data source that limits how often it can be "
        "queried. Popular tickers are cached for an hour, so trying a widely held company "
        "(for example AAPL or MSFT) will often work immediately."
    )
    st.stop()
except Exception as e:
    st.error(
        f"Something went wrong while analyzing '{ticker}'. This is usually temporary -- "
        "please try again in a moment, or try a different ticker."
    )
    # Kept out of the headline message and behind a click: a stack-trace-like
    # string in the middle of the page reads as a broken site to a visitor,
    # but it is still the first thing needed to diagnose a real bug.
    with st.expander("Technical detail"):
        st.code(f"{type(e).__name__}: {e}")
    st.stop()

bundle = analysis.bundle
info = bundle.info
dcf = analysis.dcf
score = analysis.score

if bundle.data_as_of is not None:
    st.caption(f"Financial statements as of approximately "
               f"{pd.Timestamp(bundle.data_as_of).strftime('%Y-%m-%d')} (cached up to 1 hour).")

# Price provenance is reported separately from the statement date, and in
# more detail, because the two have completely different shelf lives and
# because the price is what moves the rating. Saying "not real-time" once at
# the bottom of a report is not enough for a number this load-bearing.
_lp = analysis.live_price
if _lp is None:
    st.caption(
        "Live price unavailable right now, so the figures below use the cached quote from the "
        "data bundle, which may be up to an hour old. Every price-dependent number "
        "(DCF upside, valuation, expected return, both scores) should be read with that in mind."
    )
else:
    _bits = [f"**{fmt_money(_lp.price)}** from {_lp.source}"]
    if _lp.cross_check_price is not None:
        _gap = _lp.disagreement or 0.0
        _bits.append(
            f"cross-checked against {_lp.cross_check_source} "
            f"({fmt_money(_lp.cross_check_price)}, {_gap * 100:.2f}% apart)"
        )
    if _lp.as_of is not None:
        _bits.append(f"quoted {pd.Timestamp(_lp.as_of).strftime('%Y-%m-%d %H:%M %Z').strip()}")
    # md_escape: these strings are full of money figures, and a pair of bare
    # "$" makes Streamlit render everything between them as a LaTeX math span.
    st.caption("Price: " + md_escape(" — ".join(_bits))
               + ". Refreshed at most every 60 seconds; still delayed, not a live tick.")

    if _lp.sources_disagree:
        # Two independent providers disagreeing by more than a rounding
        # difference means at least one is wrong, and there is no way here to
        # tell which. Saying so is the only honest option: quietly picking
        # the first would hide a known defect behind a confident number.
        st.warning(
            md_escape(
                f"**The two price sources disagree.** {_lp.source} reports "
                f"{fmt_money(_lp.price)} while {_lp.cross_check_source} reports "
                f"{fmt_money(_lp.cross_check_price)} — a gap of "
                f"{(_lp.disagreement or 0) * 100:.2f}%. The report below uses "
                f"{_lp.source}."
            )
            + " Because most of the Fundamental Investment Score is price-dependent, "
              "treat every valuation figure here as provisional until the "
              "discrepancy resolves.",
            icon="⚠️",
        )
    elif _lp.is_stale:
        st.caption(
            f"This quote is {_lp.age} old — normal outside market hours, when the most recent "
            "real trade is the previous session's close."
        )

# ---------------------------------------------------------------------------
# Top summary block
# ---------------------------------------------------------------------------
company_name = info.get("shortName") or info.get("longName") or ticker
st.header(f"{company_name} ({ticker})")

top1, top2, top3 = st.columns(3)
with top1:
    st.metric("Current Price", fmt_money(dcf.current_price))
with top2:
    st.metric("DCF Intrinsic Value", fmt_money(dcf.intrinsic_value_per_share))
with top3:
    upside_display = fmt_pct(dcf.upside) if not np.isnan(dcf.upside) else "N/A"
    st.metric("Potential Upside/Downside", upside_display,
              delta=upside_display if not np.isnan(dcf.upside) else None)

score_col, gauge_col = st.columns([1, 1])
with score_col:
    st.markdown(f"## Fundamental Investment Score: **{score.total:.0f} / 1000**")
    st.markdown(f"# {score.rating.upper()}")
    st.caption(
        "This measures whether buying the stock AT TODAY'S PRICE is attractive -- "
        "not just whether the company is a good business. See the four readouts below."
    )
with gauge_col:
    st.plotly_chart(charts.investment_score_gauge(score.total, score.rating), use_container_width=True)

q1, q2, q3, q4 = st.columns(4)
q1.metric("Business Quality", f"{score.business_quality_score:.0f} / 100",
          help="Growth + Profitability + Financial Health + Cash Flow Quality. Price-independent -- does not change with the stock price.")
q2.metric("Valuation Attractiveness", f"{score.valuation_attractiveness_score:.0f} / 100",
          help="Margin of safety vs. DCF, comps, own valuation history, and peers, plus growth-adjusted valuation (PEG-style). Price-dependent.")
q3.metric("Expected Return", f"{score.expected_return_score:.0f} / 100",
          help="Expected annualized return from today's price (yield + growth + multiple reversion - dilution), plus probability-weighted scenario upside. Price-dependent.")
q4.metric("Risk", f"{score.risk_score:.0f} / 100",
          help="Valuation sensitivity, leverage, earnings/margin volatility, and data confidence. Higher = lower risk.")

st.markdown("### Why?")
if score.positive_factors or score.negative_factors:
    reason_lines = []
    for p in score.positive_factors[:3]:
        reason_lines.append(f"- ✅ Strong: {p}")
    for n in score.negative_factors[:2]:
        reason_lines.append(f"- ⚠️ Weak: {n}")
    st.markdown("\n".join(reason_lines) if reason_lines else "No standout factors identified.")
else:
    st.caption("Not enough data was available to generate standout factors.")

st.divider()

# ---------------------------------------------------------------------------
# Speculation Score -- deliberately styled differently (purple accent, rocket
# icon) from the Fundamental Investment Score above, so the two are never
# mistaken for one another at a glance. Answers a DIFFERENT question: not
# "should I buy this today" but "how much potential does this have to EXPLODE
# in value if its bullish thesis plays out" -- see src/speculative.py,
# src/themes.py, and docs/methodology.md section 9.
# ---------------------------------------------------------------------------
spec = analysis.speculative
st.markdown(
    '<div style="background-color:rgba(133,72,201,0.08); border-left:5px solid #8548c9; '
    'padding:10px 16px; border-radius:4px; margin-bottom:8px;">'
    '<span style="color:#8548c9; font-weight:700; font-size:0.8rem; letter-spacing:0.06em;">'
    '🚀 SPECULATION SCORE</span></div>',
    unsafe_allow_html=True,
)
spec_score_col, spec_gauge_col = st.columns([1, 1])
with spec_score_col:
    st.markdown(f"## Speculation Score: **{spec.total:.0f} / 1000**")
    st.markdown(f"# {spec.rating.upper()}")
    st.caption(
        "This measures how much potential this stock has to EXPLODE in value if its bullish "
        "thesis, catalysts, or emerging-industry opportunity plays out. It is deliberately "
        "independent of the Fundamental Investment Score: a company can be unprofitable and "
        "expensive today (low Fundamental Score) and still have enormous speculative upside "
        "(high Speculation Score). Both can be correct at the same time. This score is NOT a "
        "probability and NOT investment advice."
    )
with spec_gauge_col:
    st.plotly_chart(charts.speculative_score_gauge(spec.total, spec.rating), use_container_width=True)

# --- The forward-looking opportunity model, surfaced explicitly ------------
if spec.theme is not None:
    _basis_label = {
        "ticker": "explicit industry mapping",
        "keyword": "business-description keywords",
        "none": "no emerging-industry match",
    }.get(spec.theme_match_basis, spec.theme_match_basis)
    opp_a, opp_b, opp_c, opp_d = st.columns(4)
    opp_a.metric("Industry opportunity", spec.theme.name,
                 help=f"Matched by {_basis_label}. {spec.theme.source_note}")
    opp_b.metric("Est. future market (TAM)", fmt_money(spec.theme.tam_usd),
                 help="A documented ESTIMATE from src/themes.py at a mid-2030s horizon, not a fact. "
                      "Different research houses publish figures differing by several times.")
    opp_c.metric("Successful-case value", fmt_money(spec.potential_market_cap),
                 help="Potential market capitalization if this company became a major winner in its "
                      "industry: TAM x a plausible winner's share x this company's observable "
                      "positioning x a winner's EV/Sales multiple. A scenario, not a forecast.")
    opp_d.metric("Potential upside", fmt_ratio(spec.upside_multiple),
                 help="Successful-case market cap divided by today's market cap. This is the only "
                      "place price enters the Speculation Score, and it reflects SCALE (a $4T company "
                      "cannot 10x as easily as a $4B one) -- not whether the stock looks expensive.")
    if spec.secondary_themes:
        st.caption("Also matched: " + ", ".join(t.name for t in spec.secondary_themes))

_SPEC_SHORT_LABELS = {
    "TAM / Future Market Opportunity": "TAM / Market",
    "Explosive Growth Potential": "Growth",
    "Industry / Technology Optionality": "Optionality",
    "Catalyst Potential": "Catalysts",
    "5-10 Year Asymmetric Upside": "Asymmetric Upside",
    "Probability of Becoming a Major Winner": "Win Probability",
    "Competitive / First-Mover Advantage": "Competitive Edge",
    "Market Mispricing / Underappreciated Potential": "Mispricing",
}
spec_cols = st.columns(4)
for i, cat in enumerate(spec.categories):
    label = _SPEC_SHORT_LABELS.get(cat.name, cat.name)
    pct_of_100 = (cat.points / cat.max_points * 100) if cat.max_points else 0.0
    spec_cols[i % 4].metric(label, f"{pct_of_100:.0f} / 100")

conf_col, evid_col = st.columns(2)
conf_col.metric("Speculative Confidence", f"{spec.confidence:.0f} / 100",
                help="Confidence in THIS ANALYSIS (how much underlying data was available) -- "
                     "NOT confidence that the stock will go up, and NOT a probability.")
evid_col.metric("Corroborating evidence", f"{spec.evidence_strength * 100:.0f} / 100"
                if not np.isnan(spec.evidence_strength) else "N/A",
                help="How much hard financial evidence (revenue, growth, R&D, margins, capital "
                     "investment) supports the claim that this company is a genuine participant in "
                     "its industry. This gates the theme-driven categories, so a trendy keyword "
                     "alone cannot manufacture a high score.")

# --- Required explanation for high scores ---------------------------------
if spec.explanation is not None:
    exp = spec.explanation
    st.markdown("### Why this scores as a high-potential speculative opportunity")
    if exp.explosive_qualification:
        st.success(md_escape(exp.explosive_qualification))
    st.markdown(f"**1. Why the opportunity is potentially explosive** — {md_escape(exp.why_explosive)}")
    st.markdown(f"**2. The future industry / TAM driving it** — {md_escape(exp.industry_and_tam)}")
    st.markdown("**3. Catalyst types that apply to this industry**")
    for c in exp.catalysts:
        st.markdown(f"- {md_escape(c)}")
    st.caption(
        "These are the KINDS of events that reprice companies in this industry, taken from "
        "src/themes.py. This tool has no access to news or company calendars and never claims a "
        "specific catalyst is scheduled for this company."
    )
    st.markdown(f"**4. The potential upside scenario** — {md_escape(exp.upside_scenario)}")
    st.markdown(f"**5. The biggest reason the thesis could fail** — {md_escape(exp.biggest_risk)}")

spec_thesis_col, spec_risk_col = st.columns(2)
with spec_thesis_col:
    st.markdown("**Speculative Thesis**")
    if spec.thesis:
        for t in spec.thesis:
            st.markdown(f"- {t}")
    else:
        st.caption("No category scored 65/100 or above, so no standout speculative strength was identified.")
with spec_risk_col:
    st.markdown("**Key Risks**")
    if spec.risks:
        for r in spec.risks:
            st.markdown(f"- {r}")
    else:
        st.caption(
            "No category scored 35/100 or below. That is not the same as low risk: a speculative "
            "thesis rests on an estimated future market and can fail even when every measured "
            "component looks strong. See point 5 of the explanation above."
        )

st.caption(
    "⚠️ Not a prediction, not a probability, not investment advice. Market-size and industry-growth "
    "figures are documented ESTIMATES maintained in src/themes.py, where every assumption can be "
    "inspected and edited. This score deliberately does NOT penalize a high valuation multiple "
    "(that belongs to the Fundamental Score) and deliberately does NOT reward volatility, beta, "
    "short interest, or retail popularity. See docs/methodology.md section 9."
)

with st.expander("🚀 Speculation Score Breakdown"):
    spec_rows = [{"Category": cat.name, "Points": f"{cat.points:.0f} / {cat.max_points:.0f}"} for cat in spec.categories]
    st.table(pd.DataFrame(spec_rows).set_index("Category"))
    st.markdown(f"**TOTAL: {spec.total:.0f} / 1000 -- {spec.rating}**")

    for cat in spec.categories:
        with st.expander(f"{cat.name}: {cat.points:.0f} / {cat.max_points:.0f}"):
            sub_rows = [{"Metric": s.label, "Points": f"{s.points:.1f} / {s.max_points:.0f}",
                         "Value": fmt_subscore_value(s)} for s in cat.subscores]
            st.table(pd.DataFrame(sub_rows).set_index("Metric"))

    if spec.theme is not None:
        st.markdown(
            f"**Industry assumptions used ({spec.theme.name}):** estimated TAM "
            f"{md_escape(fmt_money(spec.theme.tam_usd))}, industry growth {fmt_pct(spec.theme.industry_cagr)}, "
            f"commercial maturity {fmt_pct(spec.theme.commercial_maturity)}, breakthrough potential "
            f"{fmt_pct(spec.theme.breakthrough_potential)}, plausible winner's share of market "
            f"{fmt_pct(spec.theme.plausible_winner_share)}, winner's EV/Sales "
            f"{fmt_ratio(spec.theme.winner_ev_sales)}."
        )
        st.caption(f"Source note: {spec.theme.source_note}")

    if spec.probability_weighted_value is not None and not np.isnan(spec.probability_weighted_value) and spec.current_price:
        prob_upside = spec.probability_weighted_value / spec.current_price - 1
        st.markdown(
            f"**Probability-weighted DCF scenario value (illustrative estimate):** "
            f"{md_escape(fmt_money(spec.probability_weighted_value))} ({fmt_pct(prob_upside)} vs. current price)"
        )
        st.caption(
            "Uses ESTIMATED scenario probabilities (Bear 30% / Base 50% / Bull 17% / Extreme Bull "
            "3%) -- a documented model assumption, not a statistically derived probability "
            "distribution. Shown for illustration only; it is not part of the Speculation Score's "
            "point total."
        )

st.divider()

# ---------------------------------------------------------------------------
# Company overview
# ---------------------------------------------------------------------------
st.subheader("Company Overview")
o1, o2, o3, o4 = st.columns(4)
o1.metric("Sector", info.get("sector") or "N/A")
o1.metric("Industry", info.get("industry") or "N/A")
o2.metric("Market Cap", fmt_money(info.get("marketCap")))
o2.metric("Enterprise Value", fmt_money(info.get("enterpriseValue")))
o3.metric("Shares Outstanding", fmt_money(info.get("sharesOutstanding")).replace("$", ""))
o3.metric("Beta", fmt_num(info.get("beta")))
o4.metric("52-Week High", fmt_money(info.get("fiftyTwoWeekHigh")))
o4.metric("52-Week Low", fmt_money(info.get("fiftyTwoWeekLow")))

st.divider()

# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------
with st.expander("📈 Fundamental Investment Score Breakdown", expanded=True):
    rows = []
    for cat in score.categories:
        rows.append({"Category": cat.name, "Points": f"{cat.points:.0f} / {cat.max_points:.0f}"})
    st.table(pd.DataFrame(rows).set_index("Category"))
    st.markdown(f"**TOTAL: {score.total:.0f} / 1000 -- {score.rating}**")

    for cat in score.categories:
        with st.expander(f"{cat.name}: {cat.points:.0f} / {cat.max_points:.0f}"):
            sub_rows = [{"Metric": s.label, "Points": f"{s.points:.1f} / {s.max_points:.0f}",
                         "Value": fmt_subscore_value(s)} for s in cat.subscores]
            st.table(pd.DataFrame(sub_rows).set_index("Metric"))
            if any(s.missing for s in cat.subscores):
                st.caption(
                    "N/A = this metric's underlying data could not be retrieved or reliably calculated "
                    "(a neutral half-credit score was used so it neither unfairly helps nor hurts the "
                    "total). Every other value shown -- including 0% and negative values -- is the "
                    "actual computed number, not a placeholder."
                )

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("**Positive Factors**")
        for p in score.positive_factors:
            st.markdown(f"- {p}")
        if not score.positive_factors:
            st.caption("None identified.")
    with neg_col:
        st.markdown("**Negative Factors**")
        for n in score.negative_factors:
            st.markdown(f"- {n}")
        if not score.negative_factors:
            st.caption("None identified.")

with st.expander("💲 Price Sensitivity Test"):
    st.markdown(
        "Because the Fundamental Investment Score is designed to measure the attractiveness of "
        "**buying at a given price** rather than just company quality, the score should "
        "change meaningfully if the price were different -- holding every other input "
        "fixed. This recomputes the score at hypothetical prices 20%/10% above and below "
        "today's actual price."
    )
    ps = analysis.price_sensitivity.copy()
    ps_display = ps.copy()
    ps_display["Price"] = ps_display["Price"].map(fmt_money)
    ps_display["Score"] = ps_display["Score"].map(lambda x: f"{x:.0f} / 1000")
    st.table(ps_display.set_index("Change"))
    score_range = ps["Score"].max() - ps["Score"].min()
    if score_range >= 100:
        st.caption(
            f"The score swings {score_range:.0f} points across this +/-20% price range, "
            "confirming the rating is genuinely price-sensitive rather than a fixed "
            "quality assessment."
        )
    else:
        st.caption(
            f"The score only swings {score_range:.0f} points across this +/-20% price range -- "
            "for this company, most of the score is currently coming from price-independent "
            "Business Quality and Risk categories rather than valuation."
        )

# ---------------------------------------------------------------------------
# Financial performance
# ---------------------------------------------------------------------------
with st.expander("📑 Financial Performance (Income Statement)"):
    if analysis.income_df.empty:
        st.warning("Income statement data is unavailable for this ticker.")
    else:
        display_df = analysis.income_df.copy()
        display_df.index = [d.year if hasattr(d, "year") else d for d in display_df.index]
        st.dataframe(display_df.style.format({
            "revenue": fmt_money, "gross_profit": fmt_money, "operating_income": fmt_money,
            "net_income": fmt_money, "diluted_eps": lambda x: fmt_money(x),
            "revenue_growth": fmt_pct, "gross_margin": fmt_pct, "operating_margin": fmt_pct,
            "net_margin": fmt_pct, "eps_growth": fmt_pct,
        }), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.revenue_chart(analysis.income_df), use_container_width=True)
        c2.plotly_chart(charts.revenue_growth_chart(analysis.income_df), use_container_width=True)
        c1.plotly_chart(charts.operating_income_chart(analysis.income_df), use_container_width=True)
        c2.plotly_chart(charts.net_income_chart(analysis.income_df), use_container_width=True)
        c1.plotly_chart(charts.eps_chart(analysis.income_df), use_container_width=True)
        c2.plotly_chart(charts.operating_margin_chart(analysis.income_df), use_container_width=True)
        st.plotly_chart(charts.net_margin_chart(analysis.income_df), use_container_width=True)

        if analysis.insights:
            st.markdown("**What the trends indicate:**")
            for insight in analysis.insights:
                st.markdown(f"- {insight}")

# ---------------------------------------------------------------------------
# Cash flow
# ---------------------------------------------------------------------------
with st.expander("💵 Free Cash Flow Analysis"):
    if analysis.fcf_df.empty:
        st.warning("Cash flow data is unavailable for this ticker.")
    elif is_financial_services_company(analysis.bundle):
        st.warning(
            "⚠️ Free Cash Flow is not a meaningful metric for banks, insurers, broker-dealers, "
            "asset managers, and other financial-services companies. Their reported Operating Cash "
            "Flow includes loan origination/sale, deposit, and premium activity -- a core, ongoing "
            "part of the business, not a discretionary item the way CapEx is for an industrial "
            "company. FCF, FCF margin, and FCF growth are shown as N/A rather than a confidently "
            "wrong number; Operating Cash Flow and CapEx are still shown below as reported."
        )
        display_df = analysis.fcf_df.copy()
        display_df.index = [d.year if hasattr(d, "year") else d for d in display_df.index]
        st.dataframe(display_df.style.format({
            "operating_cash_flow": fmt_money, "capex": fmt_money, "free_cash_flow": fmt_money,
            "fcf_margin": fmt_pct, "fcf_growth": fmt_pct,
        }), use_container_width=True)
    else:
        st.markdown(
            "Free cash flow (FCF = Operating Cash Flow - CapEx) represents the cash a business "
            "generates after reinvesting in itself -- it is the cash available to lenders and "
            "shareholders, and is the direct input to the DCF valuation below."
        )
        display_df = analysis.fcf_df.copy()
        display_df.index = [d.year if hasattr(d, "year") else d for d in display_df.index]
        st.dataframe(display_df.style.format({
            "operating_cash_flow": fmt_money, "capex": fmt_money, "free_cash_flow": fmt_money,
            "fcf_margin": fmt_pct, "fcf_growth": fmt_pct,
        }), use_container_width=True)
        st.plotly_chart(charts.fcf_chart(analysis.fcf_df), use_container_width=True)

        if not analysis.earnings_conversion.empty:
            latest_conv = analysis.earnings_conversion.dropna()
            if not latest_conv.empty:
                st.metric("Latest Earnings-to-Cash Conversion (OCF / Net Income)",
                          fmt_ratio(latest_conv.iloc[-1]))

# ---------------------------------------------------------------------------
# Balance sheet / financial health
# ---------------------------------------------------------------------------
with st.expander("🏦 Balance Sheet & Financial Health"):
    if analysis.balance_df.empty:
        st.warning("Balance sheet data is unavailable for this ticker.")
    else:
        display_df = analysis.balance_df.copy()
        display_df.index = [d.year if hasattr(d, "year") else d for d in display_df.index]
        st.dataframe(display_df.style.format({
            "cash": fmt_money, "total_debt": fmt_money, "current_assets": fmt_money,
            "current_liabilities": fmt_money, "total_assets": fmt_money, "total_equity": fmt_money,
            "net_debt": fmt_money, "current_ratio": fmt_ratio, "debt_to_equity": fmt_ratio,
        }), use_container_width=True)

    st.markdown("**Credit / Leverage Metrics (latest fiscal year)**")
    cm1, cm2, cm3 = st.columns(3)
    cm1.metric("Net Debt / EBITDA", fmt_ratio(analysis.credit.get("net_debt_to_ebitda")))
    cm2.metric("Debt / EBITDA", fmt_ratio(analysis.credit.get("debt_to_ebitda")))
    cm3.metric("Interest Coverage", fmt_ratio(analysis.credit.get("interest_coverage")))

# ---------------------------------------------------------------------------
# ROE / ROIC
# ---------------------------------------------------------------------------
with st.expander("🎯 Capital Efficiency (ROE & ROIC)"):
    st.markdown(
        "**ROE** = Net Income / Average Shareholders' Equity. "
        "**ROIC** = NOPAT / Average Invested Capital, where NOPAT = EBIT x (1 - Effective Tax Rate) "
        "and Invested Capital = Total Debt + Total Equity - Cash & Equivalents. "
        "ROIC strips out financing mix and excess cash, making it a cleaner measure of how "
        "efficiently the underlying business uses capital than ROE alone."
    )
    rc1, rc2 = st.columns(2)
    if not analysis.roe.dropna().empty:
        rc1.plotly_chart(charts.roe_chart(analysis.roe), use_container_width=True)
    else:
        rc1.warning("ROE could not be calculated (insufficient data).")
    if not analysis.roic.dropna().empty:
        rc2.plotly_chart(charts.roic_chart(analysis.roic), use_container_width=True)
    else:
        rc2.warning("ROIC could not be calculated (insufficient data).")

# ---------------------------------------------------------------------------
# DCF valuation
# ---------------------------------------------------------------------------
with st.expander("🧮 DCF Valuation", expanded=True):
    st.markdown("**Weighted Average Cost of Capital (WACC)**")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Risk-Free Rate", fmt_pct(analysis.assumptions.risk_free_rate.value))
    w1.metric("Beta", fmt_num(analysis.assumptions.beta.value))
    w2.metric("Equity Risk Premium", fmt_pct(analysis.assumptions.equity_risk_premium.value))
    w2.metric("Cost of Equity", fmt_pct(analysis.wacc.cost_of_equity))
    w3.metric("Pre-Tax Cost of Debt", fmt_pct(analysis.wacc.pretax_cost_of_debt))
    w3.metric("After-Tax Cost of Debt", fmt_pct(analysis.wacc.after_tax_cost_of_debt))
    w4.metric("Equity Weight", fmt_pct(analysis.wacc.equity_value_weight))
    w4.metric("Debt Weight", fmt_pct(analysis.wacc.debt_value_weight))
    st.metric("WACC", fmt_pct(analysis.wacc.wacc))

    st.markdown("**Forecast Projection**")
    proj_display = analysis.dcf.projection.copy()
    st.dataframe(proj_display.style.format({
        "revenue": fmt_money, "revenue_growth": fmt_pct, "ebit": fmt_money, "operating_margin": fmt_pct,
        "tax_rate": fmt_pct, "nopat": fmt_money, "d_and_a": fmt_money, "capex": fmt_money,
        "change_in_nwc": fmt_money, "free_cash_flow": fmt_money, "present_value_fcf": fmt_money,
    }), use_container_width=True)
    st.plotly_chart(charts.projected_fcf_chart(analysis.dcf.projection), use_container_width=True)

    st.markdown("**Valuation Bridge**")
    b1, b2, b3 = st.columns(3)
    b1.metric("PV of Forecast FCF", fmt_money(analysis.dcf.pv_forecast_fcf))
    b1.metric("Terminal Value (Gordon Growth)", fmt_money(analysis.dcf.terminal_value_gordon))
    b2.metric("PV of Terminal Value", fmt_money(analysis.dcf.pv_terminal_value))
    b2.metric("Enterprise Value", fmt_money(analysis.dcf.enterprise_value))
    b3.metric("Net Debt", fmt_money(analysis.dcf.net_debt))
    b3.metric("Equity Value", fmt_money(analysis.dcf.equity_value))

    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("Current Stock Price", fmt_money(analysis.dcf.current_price))
    r2.metric("DCF Intrinsic Value / Share", fmt_money(analysis.dcf.intrinsic_value_per_share))
    r3.metric("Potential Upside / Downside", fmt_pct(analysis.dcf.upside))
    st.caption(
        "The DCF is one of several inputs into the overall Fundamental Investment Score below -- "
        "it is not, by itself, a buy or sell recommendation."
    )

# ---------------------------------------------------------------------------
# Comparable companies
# ---------------------------------------------------------------------------
with st.expander("🏢 Comparable Company Analysis"):
    comps = analysis.comps
    if comps.peer_sector:
        st.markdown(f"Peers selected from the **{comps.peer_sector}** sector: {', '.join(comps.peer_tickers) or 'N/A'}")
        if comps.sector and comps.peer_sector != comps.sector:
            st.caption(
                f"⚠️ The data provider classifies this company's sector as \"{comps.sector}\", but its own "
                f"reported business description matches \"{comps.peer_sector}\" more closely -- peers were "
                f"selected from the latter to avoid comparing this company against an unrelated industry."
            )
    if comps.peer_table.empty:
        st.warning("No peer data could be retrieved for this company's sector.")
    else:
        st.dataframe(comps.peer_table.style.format({
            "P/E": fmt_ratio, "EV/EBITDA": fmt_ratio, "EV/Revenue": fmt_ratio, "P/S": fmt_ratio,
        }), use_container_width=True)
        st.markdown("**Peer Mean / Median Multiples**")
        summary_df = pd.DataFrame({"Mean": comps.mean_multiples, "Median": comps.median_multiples})
        st.table(summary_df.style.format(fmt_ratio))

        st.plotly_chart(charts.comparable_multiples_chart(comps.peer_table, ticker, comps.implied_values),
                         use_container_width=True)

        st.markdown("**Implied Valuation from Peer Multiples**")
        implied_rows = [{"Method": k, "Implied Value / Share": fmt_money(v)} for k, v in comps.implied_values.items()]
        if implied_rows:
            st.table(pd.DataFrame(implied_rows).set_index("Method"))
            st.metric("Comparable-Company Valuation Range",
                      f"{fmt_money(comps.comps_valuation_low)} - {fmt_money(comps.comps_valuation_high)}")
        else:
            st.caption("Implied valuation could not be computed (insufficient peer or target data).")

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
with st.expander("🎭 Bull / Base / Bear Scenarios"):
    st.markdown(
        "Each scenario re-runs the full DCF model with systematically adjusted growth, margin, "
        "WACC, and terminal-growth assumptions (see docs/methodology.md for the exact deltas)."
    )
    scen_rows = []
    for name in ["bear", "base", "bull"]:
        r = analysis.scenarios[name].dcf
        scen_rows.append({
            "Scenario": name.capitalize(),
            "Intrinsic Value": fmt_money(r.intrinsic_value_per_share),
            "Upside / Downside": fmt_pct(r.upside),
        })
    st.table(pd.DataFrame(scen_rows).set_index("Scenario"))
    st.plotly_chart(charts.scenario_chart(analysis.scenarios), use_container_width=True)

# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
with st.expander("🔬 Sensitivity Analysis"):
    st.markdown("**Intrinsic Value per Share: WACC vs. Terminal Growth Rate**")
    st.plotly_chart(charts.sensitivity_heatmap(analysis.wacc_grid, "WACC vs. Terminal Growth"),
                     use_container_width=True)
    st.markdown("**Intrinsic Value per Share: Year-1 Revenue Growth vs. Operating Margin**")
    st.plotly_chart(charts.sensitivity_heatmap(analysis.growth_margin_grid, "Revenue Growth vs. Operating Margin"),
                     use_container_width=True)
    st.caption(
        "These grids show how much the DCF output depends on the two assumptions with the "
        "largest impact on valuation. A wide range of outcomes signals a valuation that is "
        "highly sensitive to inputs that cannot be forecast with precision."
    )

# ---------------------------------------------------------------------------
# Risks
# ---------------------------------------------------------------------------
with st.expander("⚠️ Risks"):
    risk_cat = next(c for c in score.categories if c.name == "Risk")
    st.markdown(f"Risk score: **{risk_cat.points:.0f} / {risk_cat.max_points:.0f}** (higher = lower risk).")
    for s in risk_cat.subscores:
        frac = s.points / s.max_points if s.max_points else 0
        flag = "🟢" if frac >= 0.66 else ("🟡" if frac >= 0.33 else "🔴")
        detail = "data unavailable, neutral score used" if s.missing else fmt_subscore_value(s)
        st.markdown(f"- {flag} {s.label}: {s.points:.1f} / {s.max_points:.0f} ({detail})")
    st.caption(
        "This tool identifies risk from quantitative signals only (valuation sensitivity, "
        "leverage, and earnings/margin volatility). It does not account for qualitative risks "
        "such as litigation, management changes, regulation, or competitive disruption."
    )

# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------
with st.expander("📋 Assumptions Used in This Model"):
    a = analysis.assumptions
    assumption_rows = [
        {"Assumption": "Year-1 Revenue Growth", "Value": fmt_pct(a.revenue_growth_path[0]),
         "Type": "Model Assumption (fades to terminal growth)", "Notes": ""},
        {"Assumption": "Operating Margin", "Value": fmt_pct(a.operating_margin_path[0]),
         "Type": "Model Assumption", "Notes": "Recency-weighted average (recent years weighted more heavily)"},
        {"Assumption": "Tax Rate", "Value": fmt_pct(a.tax_rate.value), "Type": ASSUMPTION_LABELS[a.tax_rate.kind],
         "Notes": a.tax_rate.note},
        {"Assumption": "D&A % of Revenue", "Value": fmt_pct(a.da_pct_revenue.value),
         "Type": ASSUMPTION_LABELS[a.da_pct_revenue.kind], "Notes": a.da_pct_revenue.note},
        {"Assumption": "CapEx % of Revenue", "Value": fmt_pct(a.capex_pct_revenue.value),
         "Type": ASSUMPTION_LABELS[a.capex_pct_revenue.kind], "Notes": a.capex_pct_revenue.note},
        {"Assumption": "Incremental NWC % of Revenue Change", "Value": fmt_pct(a.nwc_pct_revenue_change.value),
         "Type": ASSUMPTION_LABELS[a.nwc_pct_revenue_change.kind], "Notes": ""},
        {"Assumption": "Risk-Free Rate", "Value": fmt_pct(a.risk_free_rate.value),
         "Type": ASSUMPTION_LABELS[a.risk_free_rate.kind], "Notes": ""},
        {"Assumption": "Equity Risk Premium", "Value": fmt_pct(a.equity_risk_premium.value),
         "Type": ASSUMPTION_LABELS[a.equity_risk_premium.kind], "Notes": ""},
        {"Assumption": "Beta", "Value": fmt_num(a.beta.value), "Type": ASSUMPTION_LABELS[a.beta.kind], "Notes": ""},
        {"Assumption": "Terminal Growth Rate", "Value": fmt_pct(a.terminal_growth.value),
         "Type": ASSUMPTION_LABELS[a.terminal_growth.kind], "Notes": ""},
        {"Assumption": "Forecast Period", "Value": f"{a.forecast_years} years", "Type": "Model Assumption", "Notes": ""},
    ]
    st.table(pd.DataFrame(assumption_rows).set_index("Assumption"))
    st.caption(
        "📊 Historical Data = an observed, reported figure. 🧮 Model Assumption = derived from "
        "historical data by a documented rule. 🌐 External Assumption = a widely used market/macro "
        "input not derived from this company's own filings. Full methodology in docs/methodology.md."
    )

st.divider()
st.caption(
    "This tool is an educational financial analysis project and is not financial advice. "
    "Both scores are generated from quantitative assumptions and publicly available "
    "data and should not be treated as a recommendation to buy or sell securities."
)
