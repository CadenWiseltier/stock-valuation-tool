"""
Speculation Score: a 0-1000 forward-looking measure of EXPLOSIVE UPSIDE
POTENTIAL.

THE QUESTION THIS SCORE ANSWERS
-------------------------------
    "How much potential does this stock have to EXPLODE in value if its
     bullish thesis, catalysts, or emerging-industry opportunity plays out
     -- and how credible is that scenario?"

This is deliberately NOT the question the Fundamental Investment Score
(src/scoring.py) answers. That score asks whether the stock is attractive to
BUY AT TODAY'S PRICE given its current financial condition. The two scores
are independent by design and are EXPECTED to disagree:

    IonQ-style profile:   Fundamental 350 / Speculation 900
        Currently unprofitable and expensive by conventional metrics, but
        positioned in an emerging industry with an enormous potential
        future market and credible technological optionality.

    Coca-Cola-style:      Fundamental 850 / Speculation 200
        An excellent, durable, reasonably priced business with almost no
        realistic path to becoming several times larger.

Neither pairing is a contradiction. They are two different questions.

WHAT THIS SCORE DELIBERATELY DOES *NOT* DO
------------------------------------------
1. It does NOT penalize a high valuation multiple. A rich P/E, P/S or
   EV/EBITDA is a Fundamental Score concern and appears nowhere here. Price
   enters this score in exactly ONE way -- through market capitalization, in
   the "current market cap -> potential market cap" upside ratio, because a
   $500B company genuinely cannot 20x as easily as a $5B one. That is a
   statement about scale, not about whether the stock is expensive.

2. It does NOT reward volatility, beta, short interest, options activity, or
   retail popularity. None of those appear in any formula below. Volatility
   is not upside potential; a stock that swings 10% a day has told you
   nothing about the size of its future market. This was an explicit design
   requirement and the earlier version of this module, which scored beta and
   scenario dispersion, was removed to satisfy it.

3. It does NOT invent company-specific catalysts. This tool has no access to
   news, press releases, clinical calendars, or contract announcements. What
   it does instead is name the CATALYST ARCHETYPES that apply to a company's
   industry ("NRC design certification milestones", "Phase 3 readouts"),
   drawn from the auditable table in src/themes.py, and label them as
   industry-level catalyst types rather than scheduled company events.

HOW FORWARD-LOOKING INFORMATION ENTERS THE MODEL
------------------------------------------------
Financial statements are backward-looking and cannot size a future market.
Every forward-looking assumption used here -- market size, industry growth
rate, commercial maturity, plausible winner economics -- lives in the
inspectable knowledge base at src/themes.py, with a source note per
industry. This module contains NO industry constants of its own; it reads
them all from that table. Disagree with a TAM figure? Change it in one place
and every score updates consistently.

Crucially, theme membership alone earns very little. `theme_evidence_strength`
below requires corroborating hard financial evidence (real revenue, real
growth, real R&D spend, real gross margin, real capital investment) before a
theme's optionality is credited in full. A shell company that merely writes
"quantum" or "AI" into its business description receives a fraction of the
industry's raw potential -- satisfying the requirement that a trendy keyword
must never be sufficient.

CATEGORY WEIGHTS (must sum to exactly 1000)
    TAM / Future Market Opportunity          150
    Explosive Growth Potential               175
    Industry / Technology Optionality        150
    Catalyst Potential                       125
    5-10 Year Asymmetric Upside              150
    Probability of Becoming a Major Winner   100
    Competitive / First-Mover Advantage       75
    Market Mispricing / Underappreciated      75
    -------------------------------------------
    TOTAL                                   1000

MISSING DATA
Unavailable inputs are never treated as zero. They score neutral half-credit
and are flagged `missing=True` with `value=NaN`, exactly as in
src/scoring.py, so the score reflects the opportunity rather than the
completeness of the data feed. A separate Speculative Confidence figure
(0-100) reports how much of the analysis rested on available data.

THE SCORE IS NOT A PROBABILITY. A 900 does not mean a 90% chance of
anything. It means the combination of market size, growth potential,
optionality, catalysts and asymmetry ranks in the top tier of what this
model measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data import StockDataBundle, row
from src.data import (
    CASH as _CASH, REVENUE as _REVENUE,
    RESEARCH_AND_DEVELOPMENT as _RESEARCH_AND_DEVELOPMENT,
)
from src.dcf import _cagr
from src.scoring import SubScore, CategoryScore, _linear_score
from src.themes import SpeculativeTheme, ThemeMatch, classify_theme

MAX_SPECULATIVE_SCORE = 1000

#: Score above which a structured explanation is REQUIRED (see
#: `SpeculativeExplanation`), and above which the UI renders it prominently.
EXPLANATION_REQUIRED_ABOVE = 700
#: Score above which the company must additionally be justified explicitly as
#: an "explosive" speculative opportunity.
EXPLOSIVE_QUALIFICATION_ABOVE = 850

# Illustrative probability weights across the four DCF scenario tiers, used
# ONLY for a supplementary display figure -- never fed into any point
# formula below. A documented model assumption, not an estimated
# distribution.
SPECULATIVE_SCENARIO_PROBABILITIES = {"bear": 0.30, "base": 0.50, "bull": 0.17, "extreme_bull": 0.03}

# Rating bands. Note the deliberate asymmetry at the top: the two highest
# bands are narrow and hard to reach, so "explosive" and
# "once-in-a-generation" stay rare and therefore meaningful.
SPECULATIVE_RATING_THRESHOLDS = [
    (951, "Once-in-a-Generation Speculative Opportunity"),
    (851, "Exceptional / Explosive Speculation"),
    (751, "Extreme Speculation"),
    (601, "Very High Speculation"),
    (451, "High Speculation"),
    (301, "Moderate Speculation"),
    (151, "Low Speculation"),
    (0, "Minimal Speculative Upside"),
]


def get_speculative_rating(total_score: float) -> str:
    for threshold, label in SPECULATIVE_RATING_THRESHOLDS:
        if total_score >= threshold:
            return label
    return "Minimal Speculative Upside"


@dataclass
class SpeculativeExplanation:
    """The structured justification required for any score above 700.

    Every field is built from figures computed elsewhere in this module plus
    the industry profile in src/themes.py -- there is no free-form generated
    prose and no fabricated company-specific claim anywhere in it.
    """

    why_explosive: str
    industry_and_tam: str
    catalysts: list[str]
    upside_scenario: str
    biggest_risk: str
    #: Populated only above EXPLOSIVE_QUALIFICATION_ABOVE.
    explosive_qualification: str = ""


@dataclass
class SpeculativeScoreResult:
    total: float
    rating: str
    categories: list[CategoryScore]
    #: 0-100 confidence in THIS ANALYSIS (data availability) -- explicitly
    #: NOT a probability that the bullish thesis succeeds.
    confidence: float
    thesis: list[str]
    risks: list[str]
    probability_weighted_value: float   # supplementary, not part of the score
    current_price: float
    # --- the forward-looking opportunity model, exposed for the UI ---
    theme: SpeculativeTheme = None
    theme_match_basis: str = "none"
    secondary_themes: list[SpeculativeTheme] = field(default_factory=list)
    evidence_strength: float = float("nan")   # 0-1
    market_cap: float = float("nan")
    potential_market_cap: float = float("nan")
    upside_multiple: float = float("nan")
    explanation: Optional[SpeculativeExplanation] = None


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _latest(s) -> float:
    if s is None:
        return np.nan
    s = s.dropna() if hasattr(s, "dropna") else s
    return float(s.iloc[-1]) if hasattr(s, "empty") and not s.empty else np.nan


def _first(s) -> float:
    if s is None:
        return np.nan
    s = s.dropna() if hasattr(s, "dropna") else s
    return float(s.iloc[0]) if hasattr(s, "empty") and not s.empty else np.nan


def _log_score(value: float, low: float, high: float, max_points: float) -> tuple[float, bool]:
    """Score a positive quantity on a LOG10 scale.

    Used wherever the meaningful unit of difference is a multiple rather
    than an increment: the gap between a $50B and a $500B market matters
    enormously, while the gap between $90B and $110B is noise. Scoring these
    linearly would let one huge number saturate the scale and erase all
    discrimination below it.

    A genuinely non-positive value is REAL data (a company with no revenue,
    or a potential upside below 1x), not missing data: it sits at the bottom
    of an ascending scale, or the top of a descending one. Only NaN/None is
    treated as unavailable.
    """
    if value is None or (isinstance(value, (int, float, np.floating)) and np.isnan(value)):
        return max_points * 0.5, True
    if value <= 0:
        return (0.0 if high > low else max_points), False
    return _linear_score(float(np.log10(value)), float(np.log10(low)), float(np.log10(high)), max_points)


def _rd_intensity(bundle: StockDataBundle) -> float:
    """R&D spend as a fraction of revenue, latest fiscal year."""
    rd = row(bundle.income_stmt, *_RESEARCH_AND_DEVELOPMENT)
    revenue = row(bundle.income_stmt, *_REVENUE)
    if rd is None or revenue is None:
        return np.nan
    common = rd.index.intersection(revenue.index)
    ratio = (rd.reindex(common).abs() / revenue.reindex(common)).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratio.iloc[-1]) if not ratio.empty else np.nan


def _rd_dollars(bundle: StockDataBundle) -> float:
    rd = row(bundle.income_stmt, *_RESEARCH_AND_DEVELOPMENT)
    v = _latest(rd.abs()) if rd is not None else np.nan
    return v


def _revenue_cagr(income_df: pd.DataFrame) -> float:
    if income_df is None or income_df.empty or "revenue" not in income_df:
        return np.nan
    return _cagr(income_df["revenue"].dropna())


def _market_cap(bundle: StockDataBundle, dcf_result) -> float:
    """Market capitalization, with a price x shares fallback.

    Market cap is the ONLY channel through which price enters this score --
    a large company needs a proportionally larger future to deliver the same
    multiple. This is a scale effect, not a valuation penalty.
    """
    mc = bundle.info.get("marketCap")
    if mc:
        return float(mc)
    price = getattr(dcf_result, "current_price", None) or bundle.info.get("currentPrice")
    shares = bundle.info.get("sharesOutstanding")
    if price and shares:
        return float(price) * float(shares)
    return np.nan


# ---------------------------------------------------------------------------
# The anti-hype gate
# ---------------------------------------------------------------------------
def theme_evidence_strength(bundle: StockDataBundle, income_df: pd.DataFrame,
                            fcf_df: pd.DataFrame) -> float:
    """0-1 measure of how much HARD FINANCIAL EVIDENCE supports the idea that
    this company is a real participant in its identified industry.

    This is the model's defense against keyword-driven hype. Matching a
    theme in src/themes.py is a claim about a business description; this
    function asks the independent question "is there money actually moving
    in a way consistent with that claim?" It looks at five signals, each
    normalized to 0-1 and averaged over whichever are available:

        1. Revenue scale        -- has the company commercialized anything?
        2. Revenue growth       -- is that commercialization accelerating?
        3. R&D intensity        -- is it genuinely investing in the technology?
        4. Gross margin         -- does the product have real economics,
                                   rather than reselling someone else's?
        5. Capital investment   -- is it building physical or operating capacity?

    A company with a trendy business summary but no revenue, no R&D and no
    capital spending scores near 0 here, and the theme-derived categories
    below are scaled down accordingly. A company with real revenue growth,
    heavy R&D and strong margins scores near 1 and receives the industry's
    full optionality.

    Returns 0.5 (neutral) only when NONE of the five signals is available,
    consistent with the project-wide rule that missing data is never scored
    as zero.
    """
    signals: list[float] = []

    revenue = _latest(income_df["revenue"]) if income_df is not None and not income_df.empty and "revenue" in income_df else np.nan
    if not np.isnan(revenue):
        # $1M of revenue -> 0.0, $1B -> 1.0, on a log scale.
        signals.append(float(np.clip(np.log10(max(revenue, 1.0) / 1e6) / 3.0, 0.0, 1.0)) if revenue > 0 else 0.0)

    rev_cagr = _revenue_cagr(income_df)
    if not np.isnan(rev_cagr):
        signals.append(float(np.clip(rev_cagr / 0.40, 0.0, 1.0)))

    rd_int = _rd_intensity(bundle)
    if not np.isnan(rd_int):
        signals.append(float(np.clip(rd_int / 0.15, 0.0, 1.0)))

    gross_margin = _latest(income_df["gross_margin"]) if income_df is not None and not income_df.empty and "gross_margin" in income_df else np.nan
    if not np.isnan(gross_margin):
        signals.append(float(np.clip(gross_margin / 0.60, 0.0, 1.0)))

    capex = _latest(fcf_df["capex"]) if fcf_df is not None and not fcf_df.empty and "capex" in fcf_df else np.nan
    if not np.isnan(capex) and not np.isnan(revenue) and revenue > 0:
        signals.append(float(np.clip((capex / revenue) / 0.15, 0.0, 1.0)))

    return float(np.mean(signals)) if signals else 0.5


def potential_market_cap(theme: SpeculativeTheme, evidence: float, revenue: float) -> tuple[float, float]:
    """Estimate the market capitalization this company could reach if it
    became a major winner in its industry.

    The chain is deliberately explicit and auditable:

        potential revenue = industry TAM
                          x plausible winner's share of that TAM
                          x positioning factor
        potential market cap = potential revenue x winner's EV/Sales multiple

    `positioning factor` scales the generic "plausible winner" outcome up or
    down for THIS company, from 0.25x (a weak participant with almost no
    corroborating evidence) to 2.0x (a demonstrably dominant one), driven by
    `theme_evidence_strength`. This is what stops every company in an
    industry from sharing one identical upside estimate.

    Potential revenue is floored at current revenue -- a company's successful
    scenario does not involve shrinking.

    Returns (potential_market_cap, positioning_factor).
    """
    if theme is None:
        return np.nan, np.nan
    ev = 0.5 if (evidence is None or np.isnan(evidence)) else float(evidence)
    positioning = 0.25 + 1.75 * float(np.clip(ev, 0.0, 1.0))
    potential_revenue = theme.tam_usd * theme.plausible_winner_share * positioning
    if not np.isnan(revenue) and revenue > 0:
        potential_revenue = max(potential_revenue, revenue)
    return potential_revenue * theme.winner_ev_sales, positioning


# ---------------------------------------------------------------------------
# Category 1 -- TAM / Future Market Opportunity (150)
# ---------------------------------------------------------------------------
def _score_tam(theme: SpeculativeTheme, revenue: float) -> CategoryScore:
    """How large could this company's addressable market become?

    Three independent angles, none of which references the stock price:
    how big the future market is in absolute terms, how fast it is growing,
    and -- most importantly -- how tiny today's revenue is relative to it.
    That last measure is the heart of the category: a company doing $40M of
    revenue against a $90B future market has thousands of times more
    headroom than one already earning $50B against a $500B market.
    """
    cat = CategoryScore(name="TAM / Future Market Opportunity", max_points=150)

    headroom = (theme.tam_usd / revenue) if (theme and not np.isnan(revenue) and revenue > 0) else np.nan
    pts, missing = _log_score(headroom, low=3.0, high=2000.0, max_points=60)
    cat.subscores.append(SubScore("Future market size relative to current revenue", pts, 60, missing,
                                   value=headroom, unit="ratio"))

    cagr = theme.industry_cagr if theme else np.nan
    pts, missing = _linear_score(cagr, low=0.03, high=0.35, max_points=50)
    cat.subscores.append(SubScore("Estimated industry growth rate", pts, 50, missing,
                                   value=cagr, unit="pct"))

    # The top of this scale is $500B rather than the largest TAM in the
    # table: above roughly half a trillion dollars every market is
    # unambiguously enormous, and stretching the scale to $1T+ would flatten
    # all the focused emerging industries (quantum, eVTOL, gene editing)
    # against the handful of broad ones, discriminating on a difference that
    # carries no real information for this question.
    tam = theme.tam_usd if theme else np.nan
    pts, missing = _log_score(tam, low=20e9, high=500e9, max_points=40)
    cat.subscores.append(SubScore("Absolute size of the future market", pts, 40, missing,
                                   value=tam, unit="usd"))

    return cat


# ---------------------------------------------------------------------------
# Category 2 -- Explosive Growth Potential (175)
# ---------------------------------------------------------------------------
def _score_explosive_growth(income_df: pd.DataFrame, scenario_results: dict,
                            revenue: float) -> CategoryScore:
    """Could this business grow many times over, rather than incrementally?

    Blends what the company is ALREADY doing (actual historical revenue
    growth, real data) with what the model's own bull scenario projects, and
    adds two structural enablers of explosive growth: a small revenue base
    (10x from $50M is a far more ordinary event than 10x from $50B) and a
    high gross margin (which is what lets revenue growth convert into
    disproportionate profit growth).

    The small-base subscore is scale-driven, not size-driven: it responds to
    REVENUE, not market capitalization, so a small-cap with no revenue
    momentum gains nothing from merely being small.
    """
    cat = CategoryScore(name="Explosive Growth Potential", max_points=175)

    latest_growth = _latest(income_df["revenue_growth"]) if income_df is not None and not income_df.empty and "revenue_growth" in income_df else np.nan
    hist_cagr = _revenue_cagr(income_df)
    candidates = [g for g in (latest_growth, hist_cagr) if not np.isnan(g)]
    demonstrated_growth = max(candidates) if candidates else np.nan
    pts, missing = _linear_score(demonstrated_growth, low=0.0, high=0.60, max_points=60)
    cat.subscores.append(SubScore("Demonstrated revenue growth (actual, best of latest year / multi-year CAGR)",
                                   pts, 60, missing, value=demonstrated_growth, unit="pct"))

    bull = scenario_results.get("bull") if scenario_results else None
    growth_path = bull.dcf.assumptions.revenue_growth_path if bull is not None else []
    bull_avg = float(np.mean(growth_path)) if len(growth_path) else np.nan
    pts, missing = _linear_score(bull_avg, low=0.05, high=0.40, max_points=45)
    cat.subscores.append(SubScore("Modeled bull-case sustained growth (5-year average)", pts, 45, missing,
                                   value=bull_avg, unit="pct"))

    pts, missing = _log_score(revenue, low=50e9, high=50e6, max_points=40)
    cat.subscores.append(SubScore("Room to multiply from a small revenue base", pts, 40, missing,
                                   value=revenue, unit="usd"))

    gross_margin = _latest(income_df["gross_margin"]) if income_df is not None and not income_df.empty and "gross_margin" in income_df else np.nan
    pts, missing = _linear_score(gross_margin, low=0.10, high=0.75, max_points=30)
    cat.subscores.append(SubScore("Gross margin (scalability of growth into profit)", pts, 30, missing,
                                   value=gross_margin, unit="pct"))

    return cat


# ---------------------------------------------------------------------------
# Category 3 -- Industry / Technology Optionality (150)
# ---------------------------------------------------------------------------
def _score_optionality(theme: SpeculativeTheme, evidence: float) -> CategoryScore:
    """Does this company sit in an industry where a technological
    breakthrough could dramatically change what it is worth?

    Both subscores are industry properties from src/themes.py, and BOTH are
    gated by `theme_evidence_strength`. The gate is the explicit answer to
    "do not automatically give a high score simply because a company uses a
    trendy keyword": a company with zero corroborating financial evidence
    receives 35% of its industry's raw optionality, and only a company whose
    spending and revenue actually look like a participant receives 100%.

    The second subscore rewards EARLY commercial maturity, because an
    industry that is already settled has little left to re-rate. That is a
    measure of remaining optionality, not of quality -- the corresponding
    risk that an immature industry never commercializes is priced separately
    in "Probability of Becoming a Major Winner".
    """
    cat = CategoryScore(name="Industry / Technology Optionality", max_points=150)

    ev = 0.5 if (evidence is None or np.isnan(evidence)) else float(np.clip(evidence, 0.0, 1.0))
    gate = 0.35 + 0.65 * ev

    breakthrough = theme.breakthrough_potential if theme else np.nan
    if np.isnan(breakthrough):
        pts, missing, raw = 90 * 0.5, True, np.nan
    else:
        raw = breakthrough * gate
        pts, missing = raw * 90, False
    cat.subscores.append(SubScore("Technological breakthrough potential (industry, gated by company evidence)",
                                   pts, 90, missing, value=raw, unit="none"))

    maturity = theme.commercial_maturity if theme else np.nan
    pts, missing = _linear_score(maturity, low=0.90, high=0.10, max_points=60)
    if not missing:
        pts *= gate
    cat.subscores.append(SubScore("Remaining optionality (how early the industry still is)", pts, 60, missing,
                                   value=maturity, unit="none"))

    return cat


# ---------------------------------------------------------------------------
# Category 4 -- Catalyst Potential (125)
# ---------------------------------------------------------------------------
def _score_catalysts(theme: SpeculativeTheme, fcf_df: pd.DataFrame,
                     income_df: pd.DataFrame) -> CategoryScore:
    """How likely is a discrete event to reprice this stock sharply?

    Split into the industry's catalyst DENSITY (biotech readouts and
    regulatory approvals reprice stocks violently; consumer staples rarely
    do) and two company-specific, computable signs that something is
    actually approaching:

      * Capital investment ramp -- factories, fabs, data centers and
        launch facilities are built before they generate revenue, so a
        sharply rising capital-expenditure line is a real, measurable
        precursor to new capacity coming online.
      * Margin inflection -- a company whose operating margin is improving
        rapidly is approaching the profitability crossover, historically one
        of the most powerful single repricing events for a growth company.

    What this does NOT do is claim a specific catalyst is scheduled. The
    named archetypes surfaced in the UI come from src/themes.py and are
    presented as the catalyst TYPES that apply to the industry.
    """
    cat = CategoryScore(name="Catalyst Potential", max_points=125)

    intensity = theme.catalyst_intensity if theme else np.nan
    pts, missing = _linear_score(intensity, low=0.15, high=0.95, max_points=45)
    cat.subscores.append(SubScore("Industry catalyst density (frequency of repricing events)", pts, 45, missing,
                                   value=intensity, unit="none"))

    capex_series = fcf_df["capex"].dropna() if fcf_df is not None and not fcf_df.empty and "capex" in fcf_df else pd.Series(dtype=float)
    capex_cagr = _cagr(capex_series)
    pts, missing = _linear_score(capex_cagr, low=0.0, high=0.60, max_points=40)
    cat.subscores.append(SubScore("Capacity build-out pace (CapEx growth)", pts, 40, missing,
                                   value=capex_cagr, unit="pct"))

    margin_delta = np.nan
    if income_df is not None and not income_df.empty and "operating_margin" in income_df:
        margins = income_df["operating_margin"].dropna()
        if len(margins) >= 2:
            margin_delta = float(margins.iloc[-1] - margins.iloc[0])
    pts, missing = _linear_score(margin_delta, low=0.0, high=0.25, max_points=40)
    cat.subscores.append(SubScore("Approach to a profitability inflection (operating-margin improvement)",
                                   pts, 40, missing, value=margin_delta, unit="pct"))

    return cat


# ---------------------------------------------------------------------------
# Category 5 -- 5-10 Year Asymmetric Upside (150)
# ---------------------------------------------------------------------------
def _score_asymmetric_upside(upside_multiple: float, mcap: float, credit: dict) -> CategoryScore:
    """The single most important question: if this company becomes a major
    winner, how many times larger could it be?

    Scored on the ratio of potential market cap (see `potential_market_cap`)
    to today's market cap, on a log scale, so that 20x scores dramatically
    higher than 2x rather than marginally higher.

    The second subscore turns that into a true ASYMMETRY measure by dividing
    the potential gain by the plausible downside. Downside severity is
    reduced by net cash on the balance sheet: a company holding cash worth a
    meaningful fraction of its market value has a harder floor beneath it
    than one with none, so the same 10x upside represents a better
    risk-reward. This is why a well-funded speculative company outranks an
    identically positioned but nearly insolvent one.

    Note again what is absent: no P/E, no EV/Sales, no comparison to peers.
    Being expensive does not reduce this score. Being ALREADY LARGE does,
    because a $4T company cannot plausibly become a $40T one.
    """
    cat = CategoryScore(name="5-10 Year Asymmetric Upside", max_points=150)

    pts, missing = _log_score(upside_multiple, low=1.5, high=15.0, max_points=100)
    cat.subscores.append(SubScore("Potential upside if the bullish thesis succeeds (potential mkt cap / current)",
                                   pts, 100, missing, value=upside_multiple, unit="ratio"))

    net_debt = credit.get("net_debt", np.nan) if credit else np.nan
    downside_severity = np.nan
    if not np.isnan(mcap) and mcap > 0:
        net_cash = -net_debt if not np.isnan(net_debt) else 0.0
        downside_severity = float(np.clip(1.0 - max(net_cash, 0.0) / mcap, 0.40, 1.0))

    asymmetry = np.nan
    if not np.isnan(upside_multiple) and not np.isnan(downside_severity) and downside_severity > 0:
        asymmetry = (upside_multiple - 1.0) / downside_severity
    pts, missing = _log_score(asymmetry, low=1.0, high=20.0, max_points=50)
    cat.subscores.append(SubScore("Upside-to-downside asymmetry (gain potential vs. capital genuinely at risk)",
                                   pts, 50, missing, value=asymmetry, unit="ratio"))

    return cat


# ---------------------------------------------------------------------------
# Category 6 -- Probability of Becoming a Major Winner (100)
# ---------------------------------------------------------------------------
def _score_win_probability(bundle: StockDataBundle, income_df: pd.DataFrame,
                           fcf_df: pd.DataFrame, credit: dict,
                           dilution_rate: float, revenue: float) -> CategoryScore:
    """Speculation is not imagination. This category is the brake.

    A company with a vast theoretical opportunity but no evidence that it
    can execute must not reach the top bands on hope alone, so this category
    asks whether the bullish scenario is actually reachable from here:

      * Commercial traction -- is anyone paying for the product yet?
      * Funding adequacy    -- can it survive long enough to find out?
        (Cash runway is the single most common cause of permanent loss in
        speculative names; a great thesis is worthless after a dilutive
        rescue financing at a 70% discount.)
      * Product economics   -- do the gross margins suggest a real
        technology advantage rather than low-margin resale?
      * R&D scale in DOLLARS -- intensity says a company tries hard;
        absolute spend says whether it can realistically outbuild rivals
        who may be spending a hundred times more.
      * Dilution discipline -- serial heavy issuance means today's
        shareholders own progressively less of any eventual success.
    """
    cat = CategoryScore(name="Probability of Becoming a Major Winner", max_points=100)

    pts, missing = _log_score(revenue, low=1e6, high=2e9, max_points=25)
    cat.subscores.append(SubScore("Commercial traction (revenue actually being generated)", pts, 25, missing,
                                   value=revenue, unit="usd"))

    cash = _latest(row(bundle.balance_sheet, *_CASH))
    fcf_latest = _latest(fcf_df["free_cash_flow"]) if fcf_df is not None and not fcf_df.empty and "free_cash_flow" in fcf_df else np.nan
    runway_years = np.nan
    if not np.isnan(fcf_latest):
        if fcf_latest >= 0:
            runway_years = 10.0     # self-funding: runway is not the binding constraint
        elif not np.isnan(cash):
            runway_years = cash / abs(fcf_latest) if cash > 0 else 0.0
    pts, missing = _linear_score(runway_years, low=0.5, high=4.0, max_points=30)
    cat.subscores.append(SubScore("Funding adequacy (years of cash runway at the current burn rate)",
                                   pts, 30, missing, value=runway_years, unit="ratio"))

    gross_margin = _latest(income_df["gross_margin"]) if income_df is not None and not income_df.empty and "gross_margin" in income_df else np.nan
    pts, missing = _linear_score(gross_margin, low=0.0, high=0.65, max_points=20)
    cat.subscores.append(SubScore("Product economics (gross margin)", pts, 20, missing,
                                   value=gross_margin, unit="pct"))

    pts, missing = _log_score(_rd_dollars(bundle), low=5e6, high=2e9, max_points=15)
    cat.subscores.append(SubScore("R&D scale in absolute dollars (ability to win the technology race)",
                                   pts, 15, missing, value=_rd_dollars(bundle), unit="usd"))

    pts, missing = _linear_score(dilution_rate, low=0.35, high=0.0, max_points=10)
    cat.subscores.append(SubScore("Dilution discipline (share-count growth)", pts, 10, missing,
                                   value=dilution_rate, unit="pct"))

    return cat


# ---------------------------------------------------------------------------
# Category 7 -- Competitive / First-Mover Advantage (75)
# ---------------------------------------------------------------------------
def _score_competitive_advantage(theme: SpeculativeTheme, evidence: float,
                                 income_df: pd.DataFrame, bundle: StockDataBundle) -> CategoryScore:
    """Could this company be one of the few that actually captures the
    industry's economics, rather than one of the many that participate in it?

    Patents, network effects and switching costs cannot be read from a set
    of financial statements, so this uses the measurable shadows they cast:
    an industry structure that concentrates returns in a few winners, a
    gross margin that indicates pricing power rather than commodity
    competition, growth that OUTPACES the company's own industry (the
    cleanest available evidence of share gains, since growing slower than
    your market means losing ground however fast you grow), and sustained
    R&D intensity.
    """
    cat = CategoryScore(name="Competitive / First-Mover Advantage", max_points=75)

    ev = 0.5 if (evidence is None or np.isnan(evidence)) else float(np.clip(evidence, 0.0, 1.0))
    wtm = theme.winner_take_most if theme else np.nan
    if np.isnan(wtm):
        pts, missing, raw = 25 * 0.5, True, np.nan
    else:
        raw = wtm * (0.40 + 0.60 * ev)
        pts, missing = raw * 25, False
    cat.subscores.append(SubScore("Winner-take-most industry structure (gated by company evidence)",
                                   pts, 25, missing, value=raw, unit="none"))

    gross_margin = _latest(income_df["gross_margin"]) if income_df is not None and not income_df.empty and "gross_margin" in income_df else np.nan
    pts, missing = _linear_score(gross_margin, low=0.15, high=0.70, max_points=20)
    cat.subscores.append(SubScore("Pricing power (gross margin)", pts, 20, missing,
                                   value=gross_margin, unit="pct"))

    rev_cagr = _revenue_cagr(income_df)
    excess = (rev_cagr - theme.industry_cagr) if (theme and not np.isnan(rev_cagr)) else np.nan
    pts, missing = _linear_score(excess, low=-0.15, high=0.40, max_points=20)
    cat.subscores.append(SubScore("Market-share gains (revenue growth vs. its own industry's growth rate)",
                                   pts, 20, missing, value=excess, unit="pct"))

    rd_int = _rd_intensity(bundle)
    pts, missing = _linear_score(rd_int, low=0.02, high=0.30, max_points=10)
    cat.subscores.append(SubScore("Technology-leadership investment (R&D / revenue)", pts, 10, missing,
                                   value=rd_int, unit="pct"))

    return cat


# ---------------------------------------------------------------------------
# Category 8 -- Market Mispricing / Underappreciated Potential (75)
# ---------------------------------------------------------------------------
def _score_mispricing(theme: SpeculativeTheme, revenue: float, mcap: float,
                      potential_mcap: float, bundle: StockDataBundle) -> CategoryScore:
    """How much of the long-term opportunity is plausibly NOT yet in the price?

    This is not a valuation check and does not penalize a high multiple.
    It asks a different question: how far is today's market value from what
    the successful outcome would be worth? A company whose current market
    cap is 2% of its successful-case value has an enormous unpriced gap; one
    already at 80% has largely been paid for in advance, which is a
    statement about remaining opportunity rather than about expensiveness.

    The first subscore captures the related idea that when current revenue
    is a vanishingly small fraction of the future market, conventional
    revenue-multiple frameworks simply cannot express the opportunity, so
    the market is likely valuing the company on an outdated basis.
    """
    cat = CategoryScore(name="Market Mispricing / Underappreciated Potential", max_points=75)

    penetration_headroom = (theme.tam_usd / revenue) if (theme and not np.isnan(revenue) and revenue > 0) else np.nan
    pts, missing = _log_score(penetration_headroom, low=5.0, high=1000.0, max_points=30)
    cat.subscores.append(SubScore("Current revenue as a fraction of the future market (unmodelable optionality)",
                                   pts, 30, missing, value=penetration_headroom, unit="ratio"))

    priced_in = (mcap / potential_mcap) if (not np.isnan(mcap) and not np.isnan(potential_mcap)
                                            and potential_mcap > 0) else np.nan
    pts, missing = _linear_score(priced_in, low=0.60, high=0.02, max_points=30)
    cat.subscores.append(SubScore("Fraction of the successful-case value already in the price", pts, 30, missing,
                                   value=priced_in, unit="pct"))

    target = bundle.info.get("targetMeanPrice")
    price = bundle.info.get("currentPrice") or bundle.info.get("regularMarketPrice")
    analyst_upside = np.nan
    if target and price:
        try:
            analyst_upside = float(target) / float(price) - 1.0
        except (TypeError, ZeroDivisionError):
            analyst_upside = np.nan
    pts, missing = _linear_score(analyst_upside, low=-0.10, high=0.60, max_points=15)
    cat.subscores.append(SubScore("Analyst consensus price target vs. current price", pts, 15, missing,
                                   value=analyst_upside, unit="pct"))

    return cat


# ---------------------------------------------------------------------------
# Narrative generation -- fixed templates only, never fabricated claims
# ---------------------------------------------------------------------------
_CATEGORY_THESIS_TEMPLATES = {
    "TAM / Future Market Opportunity":
        "Operates in a large and fast-growing future market that dwarfs its current revenue base",
    "Explosive Growth Potential":
        "Demonstrated and modeled growth rates, from a small base, support a multi-fold increase in scale",
    "Industry / Technology Optionality":
        "Sits in an early-stage industry where a technological breakthrough could re-rate the whole category",
    "Catalyst Potential":
        "Operates in a catalyst-dense industry and shows measurable signs of capacity or margin inflection",
    "5-10 Year Asymmetric Upside":
        "Potential successful-case market capitalization is a large multiple of today's, with limited capital genuinely at risk",
    "Probability of Becoming a Major Winner":
        "Commercial traction, funding, margins and R&D scale support the credibility of the bullish thesis",
    "Competitive / First-Mover Advantage":
        "Growing faster than its own industry, with margins and R&D consistent with a defensible position",
    "Market Mispricing / Underappreciated Potential":
        "Today's market value reflects only a small fraction of the successful-case outcome",
}
_CATEGORY_RISK_TEMPLATES = {
    "TAM / Future Market Opportunity":
        "The addressable market is mature or already large relative to the company, limiting transformational upside",
    "Explosive Growth Potential":
        "Neither demonstrated nor modeled growth supports a multi-fold increase in scale",
    "Industry / Technology Optionality":
        "The industry is commercially settled, or financial evidence does not corroborate the technology exposure",
    "Catalyst Potential":
        "Few discrete repricing catalysts, and no measurable capacity or margin inflection underway",
    "5-10 Year Asymmetric Upside":
        "Current market capitalization is already close to a realistic successful-case value",
    "Probability of Becoming a Major Winner":
        "Weak traction, limited runway, or thin margins undermine the credibility of the bullish scenario",
    "Competitive / First-Mover Advantage":
        "Growing no faster than its industry, with little evidence of a defensible competitive position",
    "Market Mispricing / Underappreciated Potential":
        "The long-term opportunity appears largely reflected in the current price already",
}


def _generate_speculative_narrative(categories: list[CategoryScore]) -> tuple[list[str], list[str]]:
    """Build the thesis / risk bullets mechanically from each category's
    score fraction (>=65% -> thesis, <=35% -> risk), drawing text ONLY from
    the fixed dictionaries above. No company-specific claim is ever
    constructed, so the tool cannot invent a product, deal or event.
    """
    thesis, risks = [], []
    for cat in categories:
        if cat.max_points == 0:
            continue
        frac = cat.points / cat.max_points
        if frac >= 0.65 and cat.name in _CATEGORY_THESIS_TEMPLATES:
            thesis.append(_CATEGORY_THESIS_TEMPLATES[cat.name])
        elif frac <= 0.35 and cat.name in _CATEGORY_RISK_TEMPLATES:
            risks.append(_CATEGORY_RISK_TEMPLATES[cat.name])
    return thesis[:5], risks[:5]


def _fmt_usd(x: float) -> str:
    if x is None or np.isnan(x):
        return "N/A"
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= scale:
            return f"${x / scale:,.1f}{suffix}"
    return f"${x:,.0f}"


def _build_explanation(*, total: float, theme: SpeculativeTheme, match: ThemeMatch,
                       categories: list[CategoryScore], revenue: float, mcap: float,
                       potential_mcap: float, upside_multiple: float,
                       evidence: float, company_name: str) -> Optional[SpeculativeExplanation]:
    """Assemble the structured explanation required above a score of 700.

    Every sentence is a template with computed figures substituted in. There
    is no generated prose, no claim this tool has not measured, and no
    invented event. The "biggest reason the thesis could fail" is chosen
    mechanically as the weakest-scoring category, so it is always the
    model's own most negative finding rather than a token caveat.
    """
    if total < EXPLANATION_REQUIRED_ABOVE:
        return None

    strongest = sorted(categories, key=lambda c: (c.points / c.max_points) if c.max_points else 0, reverse=True)[:3]
    weakest = min(categories, key=lambda c: (c.points / c.max_points) if c.max_points else 1.0)

    penetration = f"{revenue / theme.tam_usd * 100:.2f}%" if (theme.tam_usd and not np.isnan(revenue) and revenue > 0) else "an immaterial share"

    why = (
        f"{company_name} scores {total:.0f}/1000 primarily on "
        + ", ".join(c.name for c in strongest)
        + ". The combination that produces an explosive profile is present: a future market far larger than "
          "the company's current revenue base, growth capable of compounding into a multi-fold increase in "
          "scale, and a successful-case valuation well above today's."
    )

    industry = (
        f"Industry: {theme.name}. Estimated addressable market of roughly {_fmt_usd(theme.tam_usd)} at a "
        f"mid-2030s horizon, growing at an estimated {theme.industry_cagr * 100:.0f}% per year, with commercial "
        f"maturity assessed at {theme.commercial_maturity * 100:.0f}% (lower means more of the opportunity is "
        f"still ahead). Trailing revenue of {_fmt_usd(revenue)} represents {penetration} of that market. "
        f"Classification basis: {'explicit industry mapping' if match.match_basis == 'ticker' else 'business-description keywords' if match.match_basis == 'keyword' else 'no emerging-industry match'}; "
        f"corroborating financial evidence scored {evidence * 100:.0f}/100. Market-size figures are documented "
        f"estimates from src/themes.py, not facts."
    )

    upside = (
        f"If {company_name} became a major winner in {theme.name} -- capturing roughly "
        f"{theme.plausible_winner_share * 100:.0f}% of that market, adjusted for its currently observable "
        f"positioning -- the implied successful-case market capitalization is approximately "
        f"{_fmt_usd(potential_mcap)}, against {_fmt_usd(mcap)} today: about {upside_multiple:.1f}x. "
        f"This is a scenario, not a forecast, and it assumes the industry reaches its estimated size."
    )

    # Only attach the category's risk description when the category is
    # ACTUALLY weak. Naming the lowest-scoring category is always
    # informative, but pairing a 66/100 score with a sentence beginning
    # "Few discrete repricing catalysts" states something the score itself
    # contradicts -- found while visually verifying a real high-scoring
    # company. Above the threshold, the honest statement is that no single
    # component is weak and the binding risk is the industry-level one.
    weakest_frac = (weakest.points / weakest.max_points) if weakest.max_points else 1.0
    if weakest_frac < 0.50:
        lead = (f"Weakest component: {weakest.name} at {weakest_frac * 100:.0f}/100. "
                + _CATEGORY_RISK_TEMPLATES.get(weakest.name, "") + ". ")
    else:
        lead = (f"No single component scores poorly -- the lowest is {weakest.name} at "
                f"{weakest_frac * 100:.0f}/100 -- so the binding risk is not company-specific but "
                f"structural. ")
    biggest_risk = (
        lead
        + f"An industry at {theme.commercial_maturity * 100:.0f}% commercial maturity may never reach the "
          f"{_fmt_usd(theme.tam_usd)} market size assumed above, or may reach it with a different winner. "
          f"In either case essentially none of this upside materializes, and the entire thesis rests on a "
          f"market-size estimate rather than on demonstrated results."
    )

    qualification = ""
    if total >= EXPLOSIVE_QUALIFICATION_ABOVE:
        qualification = (
            f"Qualifies as an EXPLOSIVE speculative opportunity: it clears {EXPLOSIVE_QUALIFICATION_ABOVE}/1000 "
            f"by simultaneously combining a {_fmt_usd(theme.tam_usd)} emerging market, a successful-case value "
            f"roughly {upside_multiple:.0f}x today's market capitalization, high technological optionality in an "
            f"industry only {theme.commercial_maturity * 100:.0f}% commercially mature, and enough corroborating "
            f"financial evidence ({evidence * 100:.0f}/100) that the thesis is not purely conceptual. Reaching "
            f"this band requires strength across nearly every category at once -- upside potential alone is not "
            f"sufficient. It remains a high-uncertainty outcome, and the score is NOT a probability."
        )

    return SpeculativeExplanation(
        why_explosive=why,
        industry_and_tam=industry,
        catalysts=list(theme.catalyst_archetypes),
        upside_scenario=upside,
        biggest_risk=biggest_risk,
        explosive_qualification=qualification,
    )


def probability_weighted_scenario_value(scenario_results) -> float:
    """Supplementary, ILLUSTRATIVE probability-weighted intrinsic value across
    all four DCF scenario tiers. NOT part of the Speculation Score's point
    total -- shown alongside it only to keep the "probability matters, not
    just magnitude" principle visible. Returns NaN unless every tier is
    present, rather than silently blending an incomplete set.
    """
    values = {}
    for name in SPECULATIVE_SCENARIO_PROBABILITIES:
        result = scenario_results.get(name) if scenario_results else None
        if result is None:
            return np.nan
        v = result.dcf.intrinsic_value_per_share
        if v is None or np.isnan(v):
            return np.nan
        values[name] = v
    return sum(SPECULATIVE_SCENARIO_PROBABILITIES[name] * values[name] for name in values)


def compute_speculative_score(
    *,
    bundle: StockDataBundle,
    income_df: pd.DataFrame,
    fcf_df: pd.DataFrame,
    credit: dict,
    dcf_result,
    scenario_results: dict,
    dilution_rate: float,
) -> SpeculativeScoreResult:
    """Compute the full 0-1000 Speculation Score.

    Shares raw inputs with the Fundamental Investment Score (the same
    statements, the same DCF scenarios) but interprets them through an
    entirely different lens and combines them with the forward-looking
    industry model in src/themes.py. The two scores are independent
    judgments about the same company and are expected to disagree.
    """
    match = classify_theme(bundle.ticker, bundle.info)
    theme = match.theme

    revenue = _latest(income_df["revenue"]) if income_df is not None and not income_df.empty and "revenue" in income_df else np.nan
    evidence = theme_evidence_strength(bundle, income_df, fcf_df)
    mcap = _market_cap(bundle, dcf_result)
    potential_mcap, _positioning = potential_market_cap(theme, evidence, revenue)
    upside_multiple = (potential_mcap / mcap) if (not np.isnan(mcap) and mcap > 0
                                                  and not np.isnan(potential_mcap)) else np.nan

    categories = [
        _score_tam(theme, revenue),
        _score_explosive_growth(income_df, scenario_results, revenue),
        _score_optionality(theme, evidence),
        _score_catalysts(theme, fcf_df, income_df),
        _score_asymmetric_upside(upside_multiple, mcap, credit),
        _score_win_probability(bundle, income_df, fcf_df, credit, dilution_rate, revenue),
        _score_competitive_advantage(theme, evidence, income_df, bundle),
        _score_mispricing(theme, revenue, mcap, potential_mcap, bundle),
    ]

    total = float(min(max(sum(c.points for c in categories), 0), MAX_SPECULATIVE_SCORE))
    rating = get_speculative_rating(total)

    all_subscores = [s for c in categories for s in c.subscores]
    confidence = (1.0 - sum(1 for s in all_subscores if s.missing) / len(all_subscores)) * 100 if all_subscores else 50.0

    thesis, risks = _generate_speculative_narrative(categories)

    company_name = bundle.info.get("shortName") or bundle.info.get("longName") or bundle.ticker
    explanation = _build_explanation(
        total=total, theme=theme, match=match, categories=categories, revenue=revenue,
        mcap=mcap, potential_mcap=potential_mcap, upside_multiple=upside_multiple,
        evidence=evidence, company_name=company_name,
    )

    return SpeculativeScoreResult(
        total=total,
        rating=rating,
        categories=categories,
        confidence=confidence,
        thesis=thesis,
        risks=risks,
        probability_weighted_value=probability_weighted_scenario_value(scenario_results),
        current_price=getattr(dcf_result, "current_price", np.nan),
        theme=theme,
        theme_match_basis=match.match_basis,
        secondary_themes=match.secondary,
        evidence_strength=evidence,
        market_cap=mcap,
        potential_market_cap=potential_mcap,
        upside_multiple=upside_multiple,
        explanation=explanation,
    )
