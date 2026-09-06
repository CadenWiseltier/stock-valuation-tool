"""
Investment Score: a 0-1000, PRICE-SENSITIVE composite quantitative rating.

CORE PRINCIPLE: this score measures the attractiveness of BUYING THE STOCK
AT ITS CURRENT MARKET PRICE -- not simply the quality of the underlying
company. A wonderful business trading at an unreasonable price should not
score as a Buy; a decent business trading at a genuine discount can.

To make that true in the math (not just in prose), the categories below
are split into two families:

  BUSINESS QUALITY (250 of 1000 points, price-independent)
    Growth, Profitability, Financial Health, Cash Flow Quality -- these
    measure how good the company is, and do not change if the stock price
    moves. They are capped at 250 of 1000 points specifically so that
    excellent fundamentals alone cannot carry a stock to "Strong Buy":
    see test_higher_dcf_upside_alone_does_not_guarantee_top_score and
    test_quality_alone_cannot_reach_strong_buy in tests/test_scoring.py.

  INVESTMENT ATTRACTIVENESS (750 of 1000 points, PRICE-DEPENDENT)
    Valuation & Margin of Safety (250), Expected Return at Current Price
    (200), Growth-Adjusted Valuation (100), Scenario-Weighted Risk/Reward
    (100), and Risk (100) -- these measure whether TODAY'S PRICE is a good
    deal, and are recomputed by `price_sensitivity_table()` at hypothetical
    prices to demonstrate that the rating genuinely depends on price (see
    docs/methodology.md, section 7, for the full price-sensitivity table
    the project specification asked for).

Every subcomponent uses an explicit, documented linear scoring formula --
never a black-box weight. Full detail in docs/methodology.md.

Category weights (must sum to exactly 1000):
    Valuation & Margin of Safety     250   (price-dependent)
    Expected Return at Current Price 200   (price-dependent)
    Growth-Adjusted Valuation        100   (price-dependent)
    Scenario-Weighted Risk/Reward    100   (price-dependent)
    Growth                            70   (price-independent)
    Profitability                     70   (price-independent)
    Financial Health                  70   (price-independent)
    Cash Flow Quality                 40   (price-independent)
    Risk                             100   (price-independent)
    ----------------------------------------
    TOTAL                           1000
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

MAX_SCORE = 1000

# Probability weights used to blend the Bear / Base / Bull DCF scenarios
# into a single expected value (src: _score_scenario_weighted). A 25/50/25
# triangular weighting is a standard, simple way to express "the base case
# is the single most likely outcome, but the tails are not negligible"
# without claiming false precision about the true probability distribution.
SCENARIO_PROBABILITIES = {"bear": 0.25, "base": 0.50, "bull": 0.25}

# Empirical calibration for the DCF/comps/scenario "upside" scoring bands.
#
# An earlier version of this module used a -20%/-30% floor for these
# subscores. Checked against real tickers spanning both richly-valued
# growth names (AAPL, AMZN, NVDA, TSLA) and value/cyclical names (XOM, KO,
# JNJ, PFE), a 5-year FCFF DCF routinely implies -60% to -130% "upside"
# for the former group -- not because the scoring is broken, but because a
# conservative explicit-forecast DCF is well below where the market prices
# quality/growth compounders. A -30% floor meant nearly every mega-cap
# growth stock clamped to exactly 0 on these subscores with no
# differentiation between "expensive" (-65%) and "absurdly expensive"
# (-130%). Widening the floor to -100% (buying at ~2x fair value) restores
# that differentiation while leaving the scoring formula itself untouched.
# RECALIBRATED. Measured across 49 real companies spanning blue chips,
# high-growth tech, value, banks, REITs, energy and unprofitable growth, this
# project's own base-case DCF produced a MEDIAN implied upside of -54%, with
# the 25th percentile at -96% and the 75th at +17%. In other words the model
# calls the typical listed company roughly half overvalued. That is a known,
# documented consequence of a conservative 5-year explicit FCFF forecast plus
# a Gordon terminal value discounted at a CAPM WACC -- not a discovery that
# most of the market is mispriced.
#
# The floor is therefore -1.20 rather than -1.00, which places the observed
# MEDIAN company near the middle of the scale instead of in its bottom third.
# Without this, the score's single largest input was systematically biased
# downward for exactly the durable, high-quality compounders the score is
# most often used to evaluate.
DEEP_DISCOUNT_UPSIDE_FLOOR = -1.20
DEEP_DOWNSIDE_FLOOR = -1.40  # wider floor for bear-case-only subscores,
                             # which run more negative than blended/base-case
                             # upside (observed median -69%, p10 -127%)


def _linear_score(value: float, low: float, high: float, max_points: float) -> tuple[float, bool]:
    """Map `value` linearly onto [0, max_points].

    `low` is the value that earns 0 points and `high` is the value that
    earns full points -- `low` need not be numerically less than `high`,
    which is what lets the SAME helper express both "higher is better"
    (e.g. ROIC: low=0%, high=25%) and "lower is better" (e.g. Debt/Equity:
    low=2.5, high=0.0) scoring rules. The result is clamped to [0, max_points].

    Missing data (NaN / None) returns a neutral half-credit score rather
    than zero or full credit, so a single unavailable metric does not
    unfairly punish or reward a company -- flagged via the returned bool so
    the UI can disclose which inputs were unavailable.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return max_points * 0.5, True
    if low == high:
        return max_points * 0.5, False
    frac = (value - low) / (high - low)
    frac = min(max(frac, 0.0), 1.0)
    return frac * max_points, False


@dataclass
class SubScore:
    label: str
    points: float
    max_points: float
    missing: bool = False
    # The actual underlying metric value (e.g. -0.084 for an expected
    # return of -8.4%), always populated when NOT missing -- including
    # when it is negative or exactly zero. `missing=True` and `value=NaN`
    # is the ONLY representation of "unavailable"; a genuinely negative or
    # zero result is a real, displayed value, never collapsed to 0 or "N/A".
    # `unit` tells the UI how to format it: "pct" (a fraction, displayed as
    # a percentage), "ratio" (a multiple, e.g. 1.8x), or "none" (an already
    # human-scale number, e.g. a 0-100 percentile).
    value: float = float("nan")
    unit: str = "pct"
    # `not_applicable` is a THIRD state, distinct from both "available" and
    # "missing", and the distinction matters economically:
    #
    #   missing        -- the metric is meaningful for this company, but the
    #                     data provider did not supply it. Neutral half
    #                     credit, and it reduces the Risk category's data
    #                     confidence subscore (we are less sure of the score).
    #   not_applicable -- the metric is MEANINGLESS for this kind of company.
    #                     Interest coverage on a bank, gross margin on a REIT,
    #                     or a FCFF DCF on a lender are not missing data --
    #                     they are the wrong questions. Scoring them at a flat
    #                     50% dragged every bank and REIT toward ~470 (found
    #                     in the audit: JPM had 18 of 38 subscores at neutral)
    #                     AND then penalized them a second time through data
    #                     confidence, for the crime of being a bank.
    #
    # A not-applicable subscore is instead EXCLUDED from its category and its
    # points are redistributed across the subscores that do apply (see
    # `_redistribute_not_applicable`), so the company is judged on the metrics
    # that actually describe its business. It never affects data confidence.
    not_applicable: bool = False


@dataclass
class CategoryScore:
    name: str
    max_points: float
    subscores: list[SubScore] = field(default_factory=list)

    @property
    def points(self) -> float:
        return sum(s.points for s in self.subscores)


def _redistribute_not_applicable(cat: CategoryScore) -> CategoryScore:
    """Re-weight a category so that metrics which do not apply to this kind
    of company neither help nor hurt it.

    Each not-applicable subscore is assigned the rate the company EARNED on
    the subscores that do apply, times its own max_points. Summed, that is
    algebraically identical to dropping the inapplicable metrics and
    rescaling the remainder back up to the category's full weight -- while
    keeping every row and every `X / Y` denominator on screen exactly as
    before, so the displayed breakdown is unchanged in shape.

    Worked example (Financial Health, 70 pts, a bank): interest coverage,
    Net Debt/EBITDA and Debt/EBITDA (50 pts) are meaningless for a lender,
    leaving Debt/Equity and current ratio (20 pts). If the bank earns 14 of
    those 20 (70%), the three inapplicable rows are each credited at 70% of
    their own weight, and the category lands at 70% of 70 = 49 -- the bank's
    real score on the questions that apply to it, instead of 14 + 25 (a flat
    half-credit) = 39.

    If NOTHING in the category applies, the whole category falls back to a
    neutral 50%: with no applicable evidence there is no basis to judge
    either way, and inventing one would be worse than admitting it.
    """
    applicable = [s for s in cat.subscores if not s.not_applicable]
    inapplicable = [s for s in cat.subscores if s.not_applicable]
    if not inapplicable:
        return cat
    applicable_max = sum(s.max_points for s in applicable)
    rate = (sum(s.points for s in applicable) / applicable_max) if applicable_max > 0 else 0.5
    for s in inapplicable:
        s.points = rate * s.max_points
        # Render as "N/A" and keep it out of the positive/negative factor
        # lists. An inapplicable row's points come from the rest of its
        # category rather than from its own metric, so displaying the
        # underlying number next to them would invite the reader to check one
        # against the other and find they do not correspond. The metric was
        # not used; saying so is clearer than showing a figure that did not
        # drive the score.
        s.missing = True
        s.value = float("nan")
    return cat


# ---------------------------------------------------------------------------
# Company profiles: the same number means different things in different
# businesses.
#
# The audit that motivated this found the score was asking every company the
# same 38 questions regardless of whether those questions made sense. A bank
# has no gross margin, no free cash flow in the industrial sense, and no
# meaningful EV/EBITDA; a REIT is structurally levered at 5-6x EBITDA by
# design and was being scored as though that were distress (American Tower
# scored 5/100 on Financial Health); a warehouse retailer earns 12% gross
# margins as a deliberate strategy, not as weakness (Costco scored 57/100 on
# Profitability while earning excellent returns on capital).
#
# This does NOT create separate scoring systems per sector -- the categories,
# weights and formulas are identical for everyone. It only decides (a) which
# metrics are meaningful for a given business model and (b) what a good value
# looks like for the metrics that are.
# ---------------------------------------------------------------------------
@dataclass
class CompanyProfile:
    kind: str = "standard"
    #: (low, high) scale for gross margin -- a distribution business earning
    #: 12% and a software business earning 80% are both operating normally.
    gross_margin_scale: tuple[float, float] = (0.0, 0.70)
    #: Net Debt/EBITDA that scores zero. Higher for asset-backed businesses
    #: (REITs, utilities, telecoms) where leverage is the funding model
    #: rather than a warning sign.
    leverage_zero_point: float = 6.0
    #: (zero-point, full-point) scale for Debt/Equity.
    debt_to_equity_scale: tuple[float, float] = (3.5, 0.2)
    #: Whether a LOW return on invested capital should be read as evidence of
    #: a value trap (see `_score_growth_adjusted_valuation`). True for
    #: operating businesses, where persistently sub-cost-of-capital returns
    #: signal weak competitive position or structural decline. False for
    #: asset-heavy, spread-based business models -- REITs, utilities and
    #: financials all earn structurally low returns on a large capital base
    #: by design, and penalizing them for it would mistake their business
    #: model for a deteriorating one.
    roic_is_quality_signal: bool = True
    #: Use multi-year AVERAGES rather than the latest year for margins and
    #: returns. For commodity cyclicals the latest year is often a peak or a
    #: trough, and the project spec explicitly warns about scoring a
    #: "temporary earnings peak" as though it were durable.
    normalize_through_cycle: bool = False

    @property
    def is_financial(self) -> bool:
        return self.kind == "financials"

    @property
    def is_reit(self) -> bool:
        return self.kind == "reit"


_FINANCIAL_SECTORS = {"Financial Services", "Financials"}
_REIT_SECTORS = {"Real Estate"}
_CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
_LOW_GROSS_MARGIN_INDUSTRIES = (
    "discount stores", "grocery", "food distribution", "department stores",
    "auto & truck dealerships", "wholesale", "distribution", "airlines",
    "oil & gas refining", "engineering & construction",
)


def detect_company_profile(sector: str | None, industry: str | None) -> CompanyProfile:
    """Classify a company into a scoring profile from its reported sector and
    industry. Deliberately coarse and rule-based -- the goal is to avoid
    asking obviously wrong questions, not to model every business precisely.
    """
    sector = (sector or "").strip()
    industry_l = (industry or "").strip().lower()

    if sector in _FINANCIAL_SECTORS:
        # Banks, insurers, brokers and lenders: no gross margin, no
        # industrial free cash flow, no meaningful enterprise value (deposits
        # and debt ARE the raw material), and a FCFF DCF does not apply.
        return CompanyProfile(kind="financials", leverage_zero_point=np.inf,
                               roic_is_quality_signal=False)

    if sector in _REIT_SECTORS or "reit" in industry_l:
        # Property is bought with mortgages; 5-6x Net Debt/EBITDA is a normal
        # capital structure, not distress. Interest coverage still matters
        # and is where a genuinely overlevered REIT gets caught.
        return CompanyProfile(kind="reit", gross_margin_scale=(0.30, 0.90),
                              leverage_zero_point=10.0, debt_to_equity_scale=(3.0, 0.4),
                              roic_is_quality_signal=False)

    if sector in _CYCLICAL_SECTORS:
        return CompanyProfile(kind="cyclical", gross_margin_scale=(0.0, 0.55),
                              leverage_zero_point=5.0, normalize_through_cycle=True)

    if sector == "Utilities":
        return CompanyProfile(kind="utility", gross_margin_scale=(0.0, 0.60),
                              leverage_zero_point=9.0, debt_to_equity_scale=(3.0, 0.4),
                              roic_is_quality_signal=False)

    if any(k in industry_l for k in _LOW_GROSS_MARGIN_INDUSTRIES):
        # Thin gross margins are the business model. These companies compete
        # on turns and scale, and their quality shows up in ROIC, not margin.
        return CompanyProfile(kind="low_margin_retail", gross_margin_scale=(0.0, 0.30))

    return CompanyProfile(kind="standard")


@dataclass
class InvestmentScoreResult:
    total: float
    rating: str
    categories: list[CategoryScore]
    positive_factors: list[str]
    negative_factors: list[str]
    business_quality_score: float       # 0-100, price-independent
    valuation_attractiveness_score: float  # 0-100, price-dependent
    expected_return_score: float        # 0-100, price-dependent
    risk_score: float                   # 0-100, price-independent


# Aligned to the project's stated calibration bands: 900+ exceptional,
# 800-899 very strong, 700-799 strong/attractive, 600-699 moderately
# attractive, 500-599 fair/neutral, 400-499 below average, below 400 weak.
# The label vocabulary is unchanged -- only the boundaries move -- so the
# rating displayed in the UI keeps the same seven possible values.
#
# The practical effect is to make the upper ratings harder to earn. Under the
# old boundaries a score of 760 read "Strong Buy"; it now reads "Buy", and
# "Exceptional" requires 900 rather than 850. This was deliberate: the audit
# found a mid-cap scoring 957 and several ordinary companies in the "Strong
# Buy" band, which devalues the top of the scale.
RATING_THRESHOLDS = [
    (900, "Exceptional"),
    (800, "Strong Buy"),
    (700, "Buy"),
    (600, "Moderate / Watch"),
    (500, "Hold"),
    (400, "Weak"),
    (0, "Avoid"),
]


def get_rating(total_score: float) -> str:
    """Convert a 0-1000 score to a rating label using fixed thresholds
    (see RATING_THRESHOLDS / docs/methodology.md for the rationale)."""
    for threshold, label in RATING_THRESHOLDS:
        if total_score >= threshold:
            return label
    return "Avoid"


def _relative_multiple_score(target_multiple: float, peer_median: float, max_points: float) -> tuple[float, bool, float]:
    """Score a target valuation multiple relative to its peer median.

    A target multiple at 50% of the peer median (i.e. trading at half the
    peer valuation) earns full points; a target multiple at 250% of the
    peer median (2.5x more expensive than peers) earns zero. This rewards
    relative cheapness without requiring an absolute multiple threshold,
    since "expensive" vs "cheap" is sector- and cycle-dependent.

    The zero-point floor is 250%, not the more intuitive-sounding 150%:
    checked against real tickers, richly-valued mega-caps routinely trade
    at 1.5-1.9x their own peer group's median multiple (AAPL, AMZN, WMT,
    NVDA all fell in that range in testing) -- a 150% floor meant nearly
    every one of them clamped to exactly 0 with no differentiation from a
    stock trading at a truly extreme premium (e.g. a name trading at 15x
    peer median). 250% keeps that genuinely extreme case at (or near) zero
    while giving the merely-somewhat-pricier-than-peers case partial credit.
    """
    if target_multiple is None or peer_median is None or np.isnan(target_multiple) or np.isnan(peer_median) or peer_median == 0:
        return max_points * 0.5, True, np.nan
    ratio = target_multiple / peer_median
    return _linear_score(ratio, low=2.5, high=0.5, max_points=max_points)[0], False, ratio


#: Pseudo-observations used to shrink a small-sample valuation percentile
#: toward neutral. See `_historical_valuation_percentile_score`.
_PERCENTILE_SHRINKAGE_PRIOR = 3.0


def _historical_valuation_percentile_score(current_multiple: float, historical_series: pd.Series,
                                            max_points: float) -> tuple[float, bool, float]:
    """Score how cheap/expensive the CURRENT multiple is relative to the
    company's OWN historical valuation range (as opposed to peers).

    Uses `scipy.stats.percentileofscore` to find what percentage of
    historical observations sit at or below the current multiple: a low
    percentile means today's multiple is cheap relative to the company's
    own history; a high percentile means it is rich relative to its own
    history.

    SMALL-SAMPLE SHRINKAGE. The free data tier supplies only ~4 annual
    statements, so the percentile is computed from a handful of points and
    is close to binary: the audit found that over a quarter of the companies
    tested scored EXACTLY zero here, because any stock that has re-rated
    upward at all sits above all 3-4 of its own historical observations.
    Apple and JPMorgan both scored 0.0 out of 60 on this basis. Reading
    "expensive versus its own last four years" as a total wipeout of a
    60-point subscore is not a defensible inference from four data points.

    The raw percentile is therefore shrunk toward 50 (neutral) by a factor
    of n / (n + 3), so that a 4-observation sample carries roughly 57% of
    its face weight while a long history carries nearly all of it. With four
    observations, "richer than every prior year" now scores about 21% of the
    subscore instead of 0%. Requires at least 3 observations.

    Returns (points, missing, percentile) -- the SHRUNK percentile (0-100) is
    the displayed "value", so the number on screen is the number scored.
    """
    hist = historical_series.dropna() if historical_series is not None else pd.Series(dtype=float)
    if current_multiple is None or (isinstance(current_multiple, float) and np.isnan(current_multiple)) or len(hist) < 3:
        return max_points * 0.5, True, np.nan
    raw = scipy_stats.percentileofscore(hist.values, current_multiple, kind="mean")
    n = float(len(hist))
    weight = n / (n + _PERCENTILE_SHRINKAGE_PRIOR)
    percentile = 50.0 + weight * (raw - 50.0)
    # percentile 0 (cheaper than all history) -> full points; 100 (richer
    # than all history) -> 0 points.
    return _linear_score(percentile, low=100.0, high=0.0, max_points=max_points)[0], False, percentile


#: A valuation model whose implied upside is beyond +/- this magnitude is
#: treated as a model breakdown rather than a valuation signal, and reported
#: as unavailable.
#:
#: Two real cases from the audit motivated this. On the downside, a company
#: whose base-case DCF implied -22,800% "upside" -- a division by a
#: near-zero terminal free cash flow, not a claim that the stock is 228x
#: overvalued. On the upside, AT&T's DCF valued it at $107 against a $25.68
#: market price, implying a $770B enterprise for a business with declining
#: revenue and a 7% return on capital. That single number drove FULL marks
#: on four separate subscores across two categories and carried the stock to
#: a "Strong Buy".
#:
#: Discarding the extreme tail costs nothing in the range where the model is
#: trustworthy: the scoring scale already awards full credit at +30% upside,
#: so every value between +30% and +250% scores identically anyway. The
#: cutoff only removes the regime where a five-year FCFF projection with a
#: Gordon terminal value stops being informative -- empirically the top few
#: percent of observations -- and replaces a confidently wrong number with
#: an honest "not available".
MAX_CREDIBLE_DCF_UPSIDE = 2.5


def _sanitize_upside(upside: float) -> float:
    """Return `upside` unless it is so extreme that it indicates the
    valuation model broke down rather than a real result (see
    MAX_CREDIBLE_DCF_UPSIDE). Genuine large negatives are preserved -- only
    implausible magnitudes are rejected."""
    if upside is None or np.isnan(upside):
        return np.nan
    return upside if abs(upside) <= MAX_CREDIBLE_DCF_UPSIDE else np.nan


def _derive_price_dependent_multiples(
    price: float, eps_ttm: float, ebitda_ttm: float, revenue_ttm: float,
    shares_outstanding: float, net_debt: float,
) -> tuple[float, float, float]:
    """Recompute the target company's own P/E, EV/EBITDA, and EV/Revenue
    multiples AT A GIVEN PRICE, from the company's own fundamentals rather
    than trusting a data provider's frozen-at-actual-price field.

    This is what makes the price-sensitivity test possible: every
    price-dependent subscore below is computed from these three multiples
    plus the price itself, so re-running the score at a hypothetical price
    only requires calling this function with a different `price`.
    """
    target_pe = price / eps_ttm if eps_ttm and eps_ttm > 0 else np.nan
    if shares_outstanding and not np.isnan(shares_outstanding) and shares_outstanding > 0 and not np.isnan(net_debt):
        enterprise_value = price * shares_outstanding + net_debt
    else:
        enterprise_value = np.nan
    target_ev_ebitda = enterprise_value / ebitda_ttm if (ebitda_ttm and ebitda_ttm > 0 and not np.isnan(enterprise_value)) else np.nan
    target_ev_revenue = enterprise_value / revenue_ttm if (revenue_ttm and revenue_ttm > 0 and not np.isnan(enterprise_value)) else np.nan
    return target_pe, target_ev_ebitda, target_ev_revenue


def _score_valuation_and_margin_of_safety(
    price: float, dcf_intrinsic_value: float, comps_low: float, comps_high: float,
    target_pe: float, target_ev_ebitda: float, target_ev_revenue: float,
    peer_median: pd.Series, historical_pe: pd.Series, profile: CompanyProfile,
) -> CategoryScore:
    """Valuation & Margin of Safety -- 250 points, fully price-dependent.

    Answers "how much of a discount (or premium) does today's price
    represent, by four independent methods: our own DCF, comparable
    companies, this company's own valuation history, and its peer group's
    current multiples?"

    WEIGHTING NOTE (recalibration). The DCF's share of this category was cut
    from 70 to 60 points, and the peer-multiple evidence raised from 60 to
    80, because the audit found this project's own FCFF DCF is
    systematically bearish: across 49 real companies its median implied
    "upside" was -54%, and for quality compounders it routinely reads -60%
    to -130%. That is a known property of a conservative 5-year explicit
    forecast with a Gordon terminal value, not a finding about those
    companies. Since the same DCF ALSO drives the entire Scenario-Weighted
    category, letting it dominate valuation too concentrated far too much of
    the total score in one model's conservatism. Comparable-company
    evidence, which the audit found well-centred (median ~0%), and
    peer-relative multiples now carry more of the load.
    """
    cat = CategoryScore(name="Valuation & Margin of Safety", max_points=250)

    dcf_upside = (dcf_intrinsic_value / price - 1) if (price and price > 0 and dcf_intrinsic_value and not np.isnan(dcf_intrinsic_value)) else np.nan
    dcf_upside = _sanitize_upside(dcf_upside)
    pts, missing = _linear_score(dcf_upside, low=DEEP_DISCOUNT_UPSIDE_FLOOR, high=0.30, max_points=60)
    cat.subscores.append(SubScore("DCF margin of safety (base case)", pts, 60, missing, value=dcf_upside, unit="pct",
                                   not_applicable=profile.is_financial))

    comps_mid = (comps_low + comps_high) / 2 if not np.isnan(comps_low) and not np.isnan(comps_high) else np.nan
    comps_upside = (comps_mid / price - 1) if (price and price > 0 and not np.isnan(comps_mid)) else np.nan
    comps_upside = _sanitize_upside(comps_upside)
    pts, missing = _linear_score(comps_upside, low=-0.70, high=0.60, max_points=65)
    cat.subscores.append(SubScore("Comparable-company margin of safety", pts, 65, missing, value=comps_upside, unit="pct"))

    pts, missing, percentile = _historical_valuation_percentile_score(target_pe, historical_pe, max_points=45)
    cat.subscores.append(SubScore("Valuation vs. own historical range (P/E percentile)", pts, 45, missing,
                                   value=percentile, unit="none"))

    peer_pe = peer_median.get("P/E") if peer_median is not None and not peer_median.empty else np.nan
    peer_ev_ebitda = peer_median.get("EV/EBITDA") if peer_median is not None and not peer_median.empty else np.nan
    peer_ev_revenue = peer_median.get("EV/Revenue") if peer_median is not None and not peer_median.empty else np.nan
    pts1, m1, r1 = _relative_multiple_score(target_pe, peer_pe, 30)
    pts2, m2, r2 = _relative_multiple_score(target_ev_ebitda, peer_ev_ebitda, 25)
    pts3, m3, r3 = _relative_multiple_score(target_ev_revenue, peer_ev_revenue, 25)
    cat.subscores.append(SubScore("P/E relative to peers", pts1, 30, m1, value=r1, unit="ratio"))
    # Enterprise value is not a meaningful concept for a bank or insurer:
    # debt and deposits are the raw material of the business, not a claim
    # ranking ahead of equity, so EV-based multiples are excluded rather
    # than scored at a misleading neutral.
    cat.subscores.append(SubScore("EV/EBITDA relative to peers", pts2, 25, m2, value=r2, unit="ratio",
                                   not_applicable=profile.is_financial))
    cat.subscores.append(SubScore("EV/Revenue relative to peers", pts3, 25, m3, value=r3, unit="ratio",
                                   not_applicable=profile.is_financial))

    return _redistribute_not_applicable(cat)


#: Multiple reversion is capped at this annualized rate in each direction.
#: Peer EV/EBITDA medians can differ from a target's by 50x when either side
#: has a distorted denominator, which would otherwise let a single ratio
#: swamp the entire expected-return estimate.
MAX_ANNUAL_MULTIPLE_REVERSION = 0.10


def _score_expected_return(
    price: float, fcf_ttm: float, shares_outstanding: float, avg_forecast_growth: float,
    target_ev_ebitda: float, peer_median_ev_ebitda: float, forecast_years: int,
    dilution_rate: float, op_margin_std: float, eps_growth_std: float,
    eps_ttm: float = np.nan, profile: CompanyProfile | None = None,
) -> tuple[CategoryScore, float]:
    """Expected Return at Current Price -- 200 points, fully price-dependent.

    Approximates the annualized total return an investor could expect from
    buying TODAY, decomposed the way equity analysts typically reason about
    forward returns:

        Expected Return  ~=  FCF Yield (cash generated per dollar paid today)
                            + Growth (of the underlying business)
                            + Multiple Reversion (does the market's price
                              for a dollar of EBITDA drift toward the peer
                              median over the forecast horizon?)
                            - Dilution (share-count growth that dilutes
                              each existing share's claim on future cash
                              flows)

    This means the SAME expected growth rate produces a very different
    expected return depending on today's price -- exactly the "multiple
    expansion/contraction" effect the project specification calls out
    (30% growth at 20x earnings vs. 30% growth at 100x earnings).
    """
    cat = CategoryScore(name="Expected Return at Current Price", max_points=200)
    profile = profile or CompanyProfile()

    fcf_yield = (fcf_ttm / shares_outstanding) / price if (
        price and price > 0 and shares_outstanding and not np.isnan(shares_outstanding)
        and shares_outstanding > 0 and fcf_ttm is not None and not np.isnan(fcf_ttm)
    ) else np.nan

    # CASH-RETURN YIELD FALLBACK. "Operating cash flow minus capital
    # expenditure" is not always computable. It is not a meaningful figure at
    # all for a bank, insurer or lender, whose operating cash flow embeds
    # loan origination and deposit flows, so it is deliberately masked
    # upstream (see financial_analysis.is_financial_services_company); and
    # for some companies -- REITs especially -- the data source simply does
    # not supply the capital-expenditure line.
    #
    # Either way the old behaviour was the same: the ENTIRE expected-return
    # subscore, the single largest in the whole score, fell back to a flat
    # neutral. The audit found JPMorgan and Prologis both scoring exactly
    # half marks on it for want of one input.
    #
    # Earnings yield is the standard substitute for a cash yield when free
    # cash flow is unavailable. It is used ONLY when FCF yield could not be
    # computed at all -- a company whose free cash flow is genuinely negative
    # still scores on that real negative number, never on this fallback.
    if np.isnan(fcf_yield) and price and price > 0 \
            and eps_ttm is not None and not np.isnan(eps_ttm):
        fcf_yield = eps_ttm / price

    multiple_reversion_available = (
        target_ev_ebitda and not np.isnan(target_ev_ebitda) and target_ev_ebitda > 0
        and peer_median_ev_ebitda is not None and not np.isnan(peer_median_ev_ebitda) and peer_median_ev_ebitda > 0
        and forecast_years > 0
    )
    if multiple_reversion_available:
        multiple_reversion = (peer_median_ev_ebitda / target_ev_ebitda) ** (1 / forecast_years) - 1
        # Clamped: the audit found peer/target EV/EBITDA ratios ranging from
        # 0.45x to 54x, and an unclamped 54x ratio annualized over 5 years
        # implies a +115%/yr "return from multiple reversion" that would
        # dominate every other term. Multiple reversion is a real effect but
        # a second-order one; it is not permitted to exceed +/-10% a year.
        multiple_reversion = float(np.clip(multiple_reversion,
                                           -MAX_ANNUAL_MULTIPLE_REVERSION,
                                           MAX_ANNUAL_MULTIPLE_REVERSION))
    else:
        # No reliable peer reversion target -- assumed flat (0.0), NOT
        # because the true multiple-reversion effect is zero, but because
        # there is no data-backed basis to assume it moves either way.
        # This is a real assumption standing in for missing data, so it is
        # tracked (multiple_reversion_available=False) and folds into the
        # Confidence subscore below rather than silently passing as a
        # fully-known input.
        multiple_reversion = 0.0

    dilution_available = dilution_rate is not None and not np.isnan(dilution_rate)
    dilution = dilution_rate if dilution_available else 0.0  # same reasoning as multiple_reversion above
    growth = avg_forecast_growth if avg_forecast_growth is not None and not np.isnan(avg_forecast_growth) else np.nan

    # `expected_return` is NEVER floored/clamped before scoring -- it can
    # be (and regularly is) a large negative number, and that negative
    # value is exactly what should score poorly below. Only fcf_yield and
    # growth being unavailable makes the WHOLE composite unavailable;
    # a defaulted multiple_reversion or dilution still lets it compute
    # (see the confidence penalty below for how that's disclosed instead).
    expected_return = np.nan
    any_missing = np.isnan(fcf_yield) or np.isnan(growth)
    if not any_missing:
        expected_return = fcf_yield + growth + multiple_reversion - dilution

    pts, missing = _linear_score(expected_return, low=-0.10, high=0.20, max_points=165)
    cat.subscores.append(SubScore("Expected annualized return (yield + growth + multiple reversion - dilution)",
                                   pts, 165, missing or any_missing, value=expected_return, unit="pct"))

    # RECALIBRATED from (2.0 -> 0.0) and from 50 points to 35.
    #
    # Two problems, both found in the audit. First, the old scale was so wide
    # that the metric barely discriminated: the median company scored 91% of
    # the available points and almost nothing separated a stable business
    # from a moderately volatile one. Second, at 50 points this was a quarter
    # of the entire Expected Return category, spent on a volatility measure
    # that is ALSO scored twice more in the Risk category ("Earnings
    # volatility" and "Margin volatility" are built from the same two
    # series). Reducing it to 35 and tightening the scale to (0.80 -> 0.05)
    # moves weight onto the expected return itself -- the thing the category
    # is named after -- and restores real spread: the median company now
    # scores ~83% here, the 75th percentile ~39%, the 90th ~0%.
    avg_vol = np.nanmean([v for v in [op_margin_std, eps_growth_std] if v is not None and not np.isnan(v)]) if any(
        v is not None and not np.isnan(v) for v in [op_margin_std, eps_growth_std]) else np.nan
    pts, missing = _linear_score(avg_vol, low=0.80, high=0.05, max_points=35)
    # A component (multiple reversion and/or dilution) defaulting to a flat
    # assumption for lack of data is a real reduction in how much this
    # composite can be trusted, even though the composite still computes --
    # each defaulted component costs 15% of the confidence points earned
    # (so 2 defaulted components caps this subscore at 70% of what the
    # volatility-based score alone would give), rather than silently
    # reporting full confidence in a partly-assumed number.
    num_defaulted = (0 if multiple_reversion_available else 1) + (0 if dilution_available else 1)
    pts = pts * (1.0 - 0.15 * num_defaulted)
    cat.subscores.append(SubScore("Confidence in expected return (earnings/margin volatility)", pts, 35, missing,
                                   value=avg_vol, unit="pct"))

    return cat, expected_return


def _score_growth_adjusted_valuation(target_pe: float, target_ev_ebitda: float, avg_forecast_growth: float,
                                      historical_eps_growth: float = np.nan, roic: float = np.nan,
                                      profile: CompanyProfile | None = None) -> CategoryScore:
    """Growth-Adjusted Valuation -- 100 points, fully price-dependent.

    Answers "is the price justified by the growth and the quality of that
    growth?" using two independent growth-adjusted multiples (earnings-based
    and EBITDA-based, so a company with weak or negative earnings is not
    unfairly penalized by relying on P/E-derived PEG alone).

    TWO CORRECTIONS FROM THE AUDIT.

    1. GROWTH RATE. The denominator used to be this project's conservative,
       5-year-FADING forecast REVENUE growth rate alone. For a mature
       company that number is tiny -- Apple's is under 2% -- so the PEG came
       out at 19.9 against a zero-point floor of 20.0, and Apple scored 0.3
       out of 50. Its companion subscore scored 0.0 out of 50. A business
       earning 83% returns on invested capital with a 32% operating margin
       was recorded as having literally zero growth-adjusted valuation
       appeal, which is not a defensible economic statement; it is an
       artifact of dividing by a near-zero denominator.

       The fix is to use the better of forecast revenue growth and TRAILING
       EPS growth. Earnings growth is what a PEG ratio is classically
       defined against, and for a mature company that buys back stock, EPS
       grows materially faster than revenue -- that per-share growth is real
       and accrues to the shareholder. Trailing EPS growth is capped at 30%
       so that a single recovery year off a depressed base cannot manufacture
       a flattering denominator.

    2. QUALITY ADJUSTMENT. A company compounding at a high return on
       invested capital genuinely deserves a higher multiple than one
       growing at the same rate while earning its cost of capital, because
       each dollar it reinvests creates more value. Charging both the same
       PEG hurdle treats them as identical. The ratio is therefore divided
       by a quality factor of up to 1.5x, scaled by ROIC between 10% and
       30%. This is the specific mechanism by which the score avoids
       "confusing a bad number with a bad investment" on high-multiple
       compounders -- and it cuts the other way too: a low-ROIC company gets
       no relief at all from a high multiple.

    Floors are set from the empirical distribution this project's own
    formula produces (median PEG 5.5, 90th percentile 16.1), tightened from
    20.0 to 12.0 now that the denominator is no longer artificially small.
    """
    cat = CategoryScore(name="Growth-Adjusted Valuation", max_points=100)
    profile = profile or CompanyProfile()

    forecast_pct = avg_forecast_growth * 100 if avg_forecast_growth is not None and not np.isnan(avg_forecast_growth) else np.nan
    eps_pct = (min(historical_eps_growth, 0.30) * 100) if (
        historical_eps_growth is not None and not np.isnan(historical_eps_growth)) else np.nan
    # Blended 50/50 rather than taking whichever is larger. Trailing EPS
    # growth corrects the mature-company problem described above, but on its
    # own a single recovery year off a depressed base would flatter the
    # denominator badly (AT&T's trailing EPS growth was +40%). Averaging the
    # forward revenue forecast with the trailing per-share result keeps both
    # the correction and a conservative anchor.
    # For a REIT the EPS half of that blend is the distorted half, for the
    # same reason its P/E is: depreciation on appreciating property swamps
    # reported earnings and makes EPS growth erratic and often negative even
    # while cash flow grows. Prologis showed +8.1% forecast revenue growth
    # against -4.0% EPS growth, and blending them produced a 2.1% growth
    # denominator that scored an ordinary multiple as extreme. REITs
    # therefore use the revenue forecast alone.
    candidates = [g for g in (forecast_pct, eps_pct) if not np.isnan(g)]
    if profile.is_reit and not np.isnan(forecast_pct):
        candidates = [forecast_pct]
    growth_pct = float(np.mean(candidates)) if candidates else np.nan

    # QUALITY ADJUSTMENT, SYMMETRIC IN BOTH DIRECTIONS.
    #
    # Above a 10% return on invested capital the factor rises to 1.5x, which
    # divides the ratio down: a company compounding at high returns on
    # capital genuinely warrants a higher multiple, because each reinvested
    # dollar creates more value.
    #
    # Below 10% it falls to 0.75x, which divides the ratio UP, making the
    # stock score as more expensive than its headline multiple suggests.
    # This is the specification's requirement that "a low P/E should not
    # automatically produce a high score" and that the model should ask
    # whether a cheap multiple is cheap for a reason -- weak competitive
    # position, poor capital allocation, or structural decline all show up
    # as a persistently low return on capital. A statistically cheap
    # business that earns less than its cost of capital is a value trap, not
    # a bargain, and is now scored accordingly.
    quality_factor = 1.0
    if roic is not None and not np.isnan(roic):
        floor = -0.5 if profile.roic_is_quality_signal else 0.0
        quality_factor = 1.0 + 0.5 * float(np.clip((roic - 0.10) / 0.20, floor, 1.0))

    peg = (target_pe / growth_pct / quality_factor) if (
        target_pe is not None and not np.isnan(target_pe) and target_pe > 0
        and growth_pct is not None and not np.isnan(growth_pct) and growth_pct > 0
    ) else np.nan
    pts, missing = _linear_score(peg, low=12.0, high=1.0, max_points=50)
    # A REIT's P/E is not comparable to an operating company's. Property
    # depreciation is an enormous non-cash charge that suppresses reported
    # earnings without reducing cash available to shareholders, which is
    # exactly why the industry reports FFO instead. A REIT trading at 20x FFO
    # can show a P/E near 40 and score as though it were twice as expensive
    # as it is. FFO is not available from this data source, so the
    # earnings-based ratio is excluded and the category rests on the
    # EBITDA-based one, which adds depreciation back and is therefore the
    # closest available proxy.
    cat.subscores.append(SubScore("PEG-style (P/E to forecast growth)", pts, 50, missing, value=peg, unit="ratio",
                                   not_applicable=profile.is_reit))

    ev_ebitda_to_growth = (target_ev_ebitda / growth_pct / quality_factor) if (
        target_ev_ebitda is not None and not np.isnan(target_ev_ebitda) and target_ev_ebitda > 0
        and growth_pct is not None and not np.isnan(growth_pct) and growth_pct > 0
    ) else np.nan
    pts, missing = _linear_score(ev_ebitda_to_growth, low=10.0, high=1.0, max_points=50)
    cat.subscores.append(SubScore("EV/EBITDA to forecast growth", pts, 50, missing, value=ev_ebitda_to_growth,
                                   unit="ratio", not_applicable=profile.is_financial))

    return _redistribute_not_applicable(cat)


def _score_scenario_weighted(scenario_results, price: float,
                              profile: CompanyProfile | None = None) -> tuple[CategoryScore, float]:
    """Scenario-Weighted Risk/Reward -- 100 points, fully price-dependent.

    Rather than scoring the Bull, Base, and Bear cases independently (which
    can accidentally reward a stock just for having a wide range of
    possible outcomes), this category blends them into a single
    PROBABILITY-WEIGHTED expected value using SCENARIO_PROBABILITIES
    (25% bear / 50% base / 25% bull), then scores:

      1. The probability-weighted expected upside from today's price.
      2. How severe the bear-case downside is (a big weighted upside built
         on top of a catastrophic bear case is a very different bet than
         one with a mild bear case).
      3. The upside/downside asymmetry ratio (bull-case gain vs. bear-case
         loss) -- a favorable ("convex") risk/reward earns bonus points.

    The asymmetry ratio (see below) is deliberately built from the
    DIFFERENCES between scenario values, not each scenario's upside versus
    the current price: for a company whose DCF is deeply overvalued even
    in the Bull case, "Bull upside / Bear upside" can be a ratio of two
    negative numbers, which produces a meaningless (or sign-flipped)
    result. (Bull - Base) / (Base - Bear) stays well-defined and
    economically meaningful regardless of whether the absolute scenario
    values sit above or below the current price.
    """
    # REWEIGHTED (audit). The probability-weighted upside subscore is very
    # nearly the same number as the Valuation category's "DCF margin of
    # safety" -- both are (a DCF value / price - 1), one of them merely
    # blended across three scenarios that share every assumption except a
    # handful of deltas. Measured across the test universe the two moved
    # together almost perfectly. Between them, plus bear-case severity, the
    # single base DCF was driving 155 of 1000 points, so any systematic bias
    # in the DCF propagated into a sixth of the entire score three times
    # over.
    #
    # Its weight drops from 60 to 40, and the freed points move to the
    # asymmetry subscore (15 -> 30), which is the only measure in this
    # category that is genuinely independent of the DCF's absolute level: it
    # is built from the DIFFERENCES between scenarios, so it survives even
    # when every scenario sits below the current price.
    cat = CategoryScore(name="Scenario-Weighted Risk/Reward", max_points=100)
    profile = profile or CompanyProfile()

    bear_value = scenario_results["bear"].dcf.intrinsic_value_per_share
    base_value = scenario_results["base"].dcf.intrinsic_value_per_share
    bull_value = scenario_results["bull"].dcf.intrinsic_value_per_share

    values_valid = all(v is not None and not np.isnan(v) for v in [bear_value, base_value, bull_value]) and price and price > 0

    weighted_value = np.nan
    weighted_upside = np.nan
    bear_upside = np.nan
    bull_upside = np.nan
    if values_valid:
        weighted_value = (
            SCENARIO_PROBABILITIES["bear"] * bear_value
            + SCENARIO_PROBABILITIES["base"] * base_value
            + SCENARIO_PROBABILITIES["bull"] * bull_value
        )
        weighted_upside = _sanitize_upside(weighted_value / price - 1)
        bear_upside = _sanitize_upside(bear_value / price - 1)
        bull_upside = _sanitize_upside(bull_value / price - 1)

    pts, missing = _linear_score(weighted_upside, low=DEEP_DISCOUNT_UPSIDE_FLOOR, high=0.30, max_points=40)
    cat.subscores.append(SubScore("Probability-weighted expected upside (25/50/25 bear/base/bull)", pts, 40, missing,
                                   value=weighted_upside, unit="pct", not_applicable=profile.is_financial))

    pts, missing = _linear_score(bear_upside, low=DEEP_DOWNSIDE_FLOOR, high=-0.05, max_points=30)
    cat.subscores.append(SubScore("Bear-case downside severity", pts, 30, missing, value=bear_upside, unit="pct",
                                   not_applicable=profile.is_financial))

    asymmetry = np.nan
    if values_valid:
        upside_potential = bull_value - base_value
        downside_potential = base_value - bear_value
        # Both potentials should normally be positive (Bull assumptions are
        # strictly better than Base, which is strictly better than Bear --
        # see tests/test_scenarios.py::test_bull_base_bear_ordering).
        # That can still legitimately fail for a company whose economics
        # make growth value-DESTRUCTIVE (thin-to-negative margins where
        # the incremental CapEx/working-capital a higher-growth Bull case
        # requires outweighs the incremental profit it generates -- verified
        # against a real ticker in manual testing). When the ordering
        # breaks down, the ratio is not meaningful, so it is left as a
        # neutral "missing" score rather than reported as a misleadingly
        # bad (or good) number.
        if downside_potential > 0.01 * price and upside_potential > 0:
            asymmetry = upside_potential / downside_potential
    # Scale tightened from (0.5 -> 3.0) to (1.0 -> 2.2): the observed range
    # across 40 companies was 1.08 to 2.43 with a median of 1.42, so the old
    # scale spent most of its range on values that never occur and
    # compressed every real company into a narrow band.
    pts, missing = _linear_score(asymmetry, low=1.0, high=2.2, max_points=30)
    cat.subscores.append(SubScore("Upside / downside asymmetry", pts, 30, missing, value=asymmetry, unit="ratio",
                                   not_applicable=profile.is_financial))

    return _redistribute_not_applicable(cat), weighted_upside


def _growth_quality_factor(income_df: pd.DataFrame) -> float:
    """Return a 0.65-1.00 multiplier reflecting the QUALITY of growth.

    The project specification is explicit that "30% revenue growth
    accompanied by deteriorating margins and massive dilution should not
    automatically receive an exceptional growth score." Growth bought by
    selling the product below cost, or funded by printing shares, is worth
    less than growth that drops through to the owner.

    Margin trend is the measurable half of that test here (dilution is
    handled separately and directly, by converting revenue growth to a
    per-share basis in `_score_growth`). A company whose operating margin
    expanded while it grew keeps its full growth score; one whose operating
    margin fell by 10 percentage points or more over the available history
    keeps 65% of it. Returns 1.0 when the trend cannot be computed, so
    missing data never creates a penalty.
    """
    if income_df is None or income_df.empty or "operating_margin" not in income_df:
        return 1.0
    margins = income_df["operating_margin"].dropna()
    if len(margins) < 2:
        return 1.0
    delta = float(margins.iloc[-1] - margins.iloc[0])
    # 0pp or better -> 1.00; -10pp or worse -> 0.65.
    return float(np.clip(1.0 + 0.35 * (delta / 0.10), 0.65, 1.0))


def _score_growth(income_df: pd.DataFrame, fcf_df: pd.DataFrame, forecast_growth_yr1: float,
                   dilution_rate: float = np.nan) -> CategoryScore:
    """Growth -- 70 points, price-independent (Business Quality).

    RECALIBRATED in two ways.

    PER-SHARE, NOT HEADLINE. A shareholder owns a share, not the company, so
    revenue growth is scored net of share issuance. A company growing revenue
    18% while issuing 10% more shares a year has delivered 8% to its owners,
    and the specification asks explicitly that dilution "recognize the
    negative effect on existing shareholders". Buybacks (negative dilution)
    correspondingly add. EPS and FCF-per-share growth are already per-share
    by construction.

    QUALITY-ADJUSTED. Both revenue-growth subscores are multiplied by
    `_growth_quality_factor`, so growth achieved while margins collapse
    scores below growth achieved while margins expand.

    Scale ceilings were raised from 15% to 20-25% because the old ceiling
    saturated: on the test universe every company above 15% growth received
    identical full marks, which erased the difference between a 16% grower
    and a 130% grower.
    """
    cat = CategoryScore(name="Growth", max_points=70)
    quality = _growth_quality_factor(income_df)
    net_dilution = dilution_rate if (dilution_rate is not None and not np.isnan(dilution_rate)) else 0.0

    rev_growth_hist = income_df["revenue_growth"].dropna() if income_df is not None and not income_df.empty else pd.Series(dtype=float)
    avg_rev_growth = rev_growth_hist.mean() if not rev_growth_hist.empty else np.nan
    per_share_rev_growth = (avg_rev_growth - net_dilution) * quality if not np.isnan(avg_rev_growth) else np.nan
    pts, missing = _linear_score(per_share_rev_growth, low=0.0, high=0.20, max_points=18)
    cat.subscores.append(SubScore("Historical revenue growth", pts, 18, missing,
                                   value=per_share_rev_growth, unit="pct"))

    eps_growth_hist = income_df["eps_growth"].dropna() if income_df is not None and not income_df.empty else pd.Series(dtype=float)
    avg_eps_growth = eps_growth_hist.mean() if not eps_growth_hist.empty else np.nan
    pts, missing = _linear_score(avg_eps_growth, low=-0.05, high=0.25, max_points=18)
    cat.subscores.append(SubScore("Historical EPS growth", pts, 18, missing, value=avg_eps_growth, unit="pct"))

    fcf_growth_hist = fcf_df["fcf_growth"].dropna() if fcf_df is not None and not fcf_df.empty else pd.Series(dtype=float)
    avg_fcf_growth = fcf_growth_hist.mean() if not fcf_growth_hist.empty else np.nan
    pts, missing = _linear_score(avg_fcf_growth, low=-0.10, high=0.25, max_points=10)
    cat.subscores.append(SubScore("Historical FCF growth", pts, 10, missing, value=avg_fcf_growth, unit="pct"))

    forecast_per_share = ((forecast_growth_yr1 - net_dilution) * quality
                          if (forecast_growth_yr1 is not None and not np.isnan(forecast_growth_yr1)) else np.nan)
    pts, missing = _linear_score(forecast_per_share, low=0.0, high=0.20, max_points=10)
    cat.subscores.append(SubScore("Forecast (Year-1) revenue growth", pts, 10, missing,
                                   value=forecast_per_share, unit="pct"))

    growth_std = rev_growth_hist.std() if len(rev_growth_hist) >= 2 else np.nan
    pts, missing = _linear_score(growth_std, low=0.25, high=0.01, max_points=14)
    cat.subscores.append(SubScore("Revenue growth consistency", pts, 14, missing, value=growth_std, unit="pct"))

    return cat


def _score_profitability(income_df: pd.DataFrame, roe: pd.Series, roic: pd.Series,
                          profile: CompanyProfile | None = None) -> CategoryScore:
    """Profitability -- 70 points, price-independent (Business Quality).

    RECALIBRATED. Three changes, all aimed at measuring economic quality
    rather than accounting appearance.

    ROIC 15 -> 27 POINTS. Return on invested capital is the single best
    available evidence that a business creates value rather than merely
    recycling capital, and the specification asks for it to carry meaningful
    weight. At 15 of 1000 points it previously carried less weight than the
    bear case of a DCF. It is now the largest subscore in the category, and
    is measured as the AVERAGE across available years rather than the latest
    snapshot, so a single good year does not qualify as a durable return.

    NET MARGIN 10 -> 6, ROE 15 -> 12. Net margin is largely operating margin
    after financing and tax, and ROE is ROIC amplified by leverage. Both
    overlap with metrics already scored here, and the specification warns
    against "double-counting ROIC, margins and profitability metrics if they
    are all measuring essentially the same thing." The weight moves to ROIC,
    which is the least redundant of them.

    GROSS MARGIN IS BUSINESS-MODEL RELATIVE. A 0-70% scale is calibrated for
    software. Costco earns roughly 12% gross margins by deliberate strategy
    and converts them into excellent returns on capital through inventory
    turns; the audit found it scoring 57/100 on profitability, penalized for
    executing its own model correctly. The scale now comes from the company
    profile, so a distributor is judged against distributors.
    """
    cat = CategoryScore(name="Profitability", max_points=70)
    profile = profile or CompanyProfile()

    latest = lambda s: s.dropna().iloc[-1] if s is not None and not s.dropna().empty else np.nan
    # Commodity producers earn their highest margins and returns exactly when
    # the commodity is at a peak. The specification explicitly warns against
    # scoring a "temporary earnings peak" as durable quality, so cyclicals are
    # judged on their through-cycle average instead of the latest year.
    central = (lambda s: s.dropna().mean() if s is not None and not s.dropna().empty else np.nan) \
        if profile.normalize_through_cycle else latest

    gm_low, gm_high = profile.gross_margin_scale
    gross_margin = central(income_df["gross_margin"]) if income_df is not None and not income_df.empty else np.nan
    pts, missing = _linear_score(gross_margin, low=gm_low, high=gm_high, max_points=8)
    # Banks and insurers do not report a cost of revenue, so gross margin is
    # not a defined concept for them.
    cat.subscores.append(SubScore("Gross margin", pts, 8, missing, value=gross_margin, unit="pct",
                                   not_applicable=profile.is_financial))

    op_margin = central(income_df["operating_margin"]) if income_df is not None and not income_df.empty else np.nan
    pts, missing = _linear_score(op_margin, low=-0.05, high=0.32, max_points=14)
    cat.subscores.append(SubScore("Operating margin", pts, 14, missing, value=op_margin, unit="pct",
                                   not_applicable=profile.is_financial))

    net_margin = central(income_df["net_margin"]) if income_df is not None and not income_df.empty else np.nan
    pts, missing = _linear_score(net_margin, low=-0.05, high=0.25, max_points=6)
    cat.subscores.append(SubScore("Net margin", pts, 6, missing, value=net_margin, unit="pct"))

    roe_value = central(roe)
    pts, missing = _linear_score(roe_value, low=0.0, high=0.28, max_points=12)
    cat.subscores.append(SubScore("Return on equity (ROE)", pts, 12, missing, value=roe_value, unit="pct"))

    # Averaged across all available years: a durable return on capital is the
    # point, not one flattering period.
    roic_avg = roic.dropna().mean() if roic is not None and not roic.dropna().empty else np.nan
    pts, missing = _linear_score(roic_avg, low=0.02, high=0.28, max_points=27)
    # ROIC requires a meaningful "invested capital" denominator. For a bank,
    # deposits and borrowings are operating inputs rather than invested
    # capital, so the ratio is not comparable to an industrial's -- ROE is
    # the industry-standard measure instead, and is scored above.
    cat.subscores.append(SubScore("Return on invested capital (ROIC)", pts, 27, missing, value=roic_avg, unit="pct",
                                   not_applicable=profile.is_financial))

    # Reduced 5 -> 3: the same operating-margin dispersion is scored again,
    # at higher weight, in the Risk category ("Margin volatility").
    op_margin_std = income_df["operating_margin"].dropna().std() if income_df is not None and not income_df.empty and len(income_df["operating_margin"].dropna()) >= 2 else np.nan
    pts, missing = _linear_score(op_margin_std, low=0.20, high=0.0, max_points=3)
    cat.subscores.append(SubScore("Margin stability", pts, 3, missing, value=op_margin_std, unit="pct",
                                   not_applicable=profile.is_financial))

    return _redistribute_not_applicable(cat)


#: Interest coverage reported for a company that services its debt many
#: times over. Used as the displayed value when a company has no meaningful
#: interest expense at all -- see `_score_financial_health`.
EFFECTIVELY_UNLEVERED_COVERAGE = 100.0


def _score_financial_health(balance_df: pd.DataFrame, credit: dict,
                             profile: CompanyProfile | None = None) -> CategoryScore:
    """Financial Health -- 70 points, price-independent (Business Quality).

    RECALIBRATED. The specification asks that the score not "automatically
    punish a company simply for having debt", but instead ask whether the
    debt is reasonable relative to the company's cash flow. Three fixes.

    LEVERAGE WAS COUNTED FOUR TIMES. Debt/Equity, Net Debt/EBITDA and
    Debt/EBITDA (45 points here) plus "Debt risk" in the Risk category (25
    points) meant 70 of 1000 points measured essentially one thing, and
    Debt/EBITDA is nearly the same ratio as Net Debt/EBITDA. Weight moves
    from the redundant measures toward INTEREST COVERAGE (15 -> 22), which is
    the one that actually answers the specification's question: a company
    with $10B of debt and $5B of free cash flow covers its interest many
    times over and is genuinely different from one with the same debt and
    falling cash flow, even though both may show identical Debt/Equity.

    DEBT-FREE COMPANIES WERE PENALIZED FOR HAVING NO DEBT. `credit_metrics`
    returns NaN for interest coverage when there is no interest expense to
    divide by, and NaN scored a neutral 50%. So a company with no debt at all
    -- the strongest possible position on this metric -- scored exactly half
    marks, worse than a company with real but comfortably covered debt. The
    audit found this affecting 16 of 49 companies. A company with positive
    operating profit and no meaningful interest expense now receives full
    credit, which is what the underlying fact means.

    LEVERAGE THRESHOLDS ARE STRUCTURE-AWARE. A REIT funds property with
    mortgages and runs at 5-6x Net Debt/EBITDA by design; the audit found
    American Tower scoring 5/100 here, which describes the REIT business
    model rather than American Tower. The zero-point comes from the company
    profile, and interest coverage -- which still catches a genuinely
    overlevered REIT -- carries the larger weight.
    """
    cat = CategoryScore(name="Financial Health", max_points=70)
    profile = profile or CompanyProfile()
    lev_zero = profile.leverage_zero_point
    lev_applicable = not (profile.is_financial or np.isinf(lev_zero))

    latest = lambda s: s.dropna().iloc[-1] if s is not None and not s.dropna().empty else np.nan

    de = latest(balance_df["debt_to_equity"]) if balance_df is not None and not balance_df.empty else np.nan
    # Widened from 2.5 to 3.5 for an industrial: Debt/Equity is a
    # capital-structure choice, and the observed 90th percentile across the
    # test universe was 4.5 for perfectly solvent companies. The scale comes
    # from the profile because "normal" differs enormously by business model.
    de_low, de_high = profile.debt_to_equity_scale
    pts, missing = _linear_score(de, low=de_low, high=de_high, max_points=10)
    # Debt/Equity is not a capital-adequacy measure for a bank. The reported
    # "total debt" figure excludes customer deposits -- by far a bank's
    # largest liability -- so the ratio lands between 1x and 3x and reflects
    # funding-mix accounting rather than solvency. Real bank capital strength
    # is a regulatory capital ratio (CET1), which this data source does not
    # supply. Scoring the reported ratio produced an accounting artifact in
    # both directions during calibration, so it is excluded; with no
    # applicable measure left, Financial Health falls back to a neutral 50%
    # for financials, which is the honest position rather than a confident
    # wrong one.
    cat.subscores.append(SubScore("Debt / Equity", pts, 10, missing, value=de, unit="ratio",
                                   not_applicable=profile.is_financial))

    cr = latest(balance_df["current_ratio"]) if balance_df is not None and not balance_df.empty else np.nan
    pts, missing = _linear_score(cr, low=0.6, high=2.0, max_points=10)
    # A bank's balance sheet is not split into current and non-current in the
    # industrial sense; a current ratio for a lender is not interpretable.
    cat.subscores.append(SubScore("Current ratio", pts, 10, missing, value=cr, unit="ratio",
                                   not_applicable=profile.is_financial))

    nd_ebitda = credit.get("net_debt_to_ebitda")
    pts, missing = _linear_score(nd_ebitda, low=lev_zero if lev_applicable else 6.0, high=-0.5, max_points=20)
    cat.subscores.append(SubScore("Net Debt / EBITDA", pts, 20, missing, value=nd_ebitda, unit="ratio",
                                   not_applicable=not lev_applicable))

    d_ebitda = credit.get("debt_to_ebitda")
    pts, missing = _linear_score(d_ebitda, low=(lev_zero + 1.0) if lev_applicable else 6.0, high=0.0, max_points=8)
    cat.subscores.append(SubScore("Debt / EBITDA", pts, 8, missing, value=d_ebitda, unit="ratio",
                                   not_applicable=not lev_applicable))

    coverage = credit.get("interest_coverage")
    if (coverage is None or np.isnan(coverage)) and credit.get("no_meaningful_interest_expense"):
        coverage = EFFECTIVELY_UNLEVERED_COVERAGE
    pts, missing = _linear_score(coverage, low=1.5, high=12.0, max_points=22)
    cat.subscores.append(SubScore("Interest coverage", pts, 22, missing, value=coverage, unit="ratio",
                                   not_applicable=profile.is_financial))

    return _redistribute_not_applicable(cat)


def _score_cash_flow_quality(fcf_df: pd.DataFrame, earnings_conversion: pd.Series,
                              profile: CompanyProfile | None = None) -> CategoryScore:
    """Cash Flow Quality -- 40 points, price-independent (Business Quality).

    RECALIBRATED. Weight moves from the single-year FCF growth rate (10 -> 6)
    toward the level and consistency of cash generation, because one year's
    change in free cash flow is dominated by the timing of capital spending.
    The audit found Apple scoring zero on both FCF-growth subscores for a
    single down year, while generating one of the largest and most reliable
    free cash flow streams in existence -- an outcome that says more about
    the metric's noise than about Apple.

    The specification asks that negative free cash flow be read in context.
    Level (FCF margin) and reliability (positive in every year) are the
    contextual measures and now carry 24 of the 40 points; a company
    investing through a temporary FCF dip retains credit for both, while one
    burning cash persistently fails both.

    "Operating cash flow minus capital expenditure" is not meaningful for
    banks and insurers, whose operating cash flow embeds lending and deposit
    flows, so the whole category is marked not applicable for them and their
    Business Quality is judged on the categories that do apply.
    """
    cat = CategoryScore(name="Cash Flow Quality", max_points=40)
    profile = profile or CompanyProfile()
    na = profile.is_financial

    latest = lambda s: s.dropna().iloc[-1] if s is not None and not s.dropna().empty else np.nan

    fcf_margin = latest(fcf_df["fcf_margin"]) if fcf_df is not None and not fcf_df.empty else np.nan
    pts, missing = _linear_score(fcf_margin, low=-0.02, high=0.22, max_points=14)
    cat.subscores.append(SubScore("FCF margin", pts, 14, missing, value=fcf_margin, unit="pct", not_applicable=na))

    fcf_growth = latest(fcf_df["fcf_growth"]) if fcf_df is not None and not fcf_df.empty else np.nan
    pts, missing = _linear_score(fcf_growth, low=-0.20, high=0.30, max_points=6)
    cat.subscores.append(SubScore("FCF growth", pts, 6, missing, value=fcf_growth, unit="pct", not_applicable=na))

    conversion = latest(earnings_conversion)
    # Clamped: the observed range included -36x, where a near-zero net income
    # denominator produces a meaningless ratio rather than a cash-quality signal.
    if conversion is not None and not np.isnan(conversion) and abs(conversion) > 10.0:
        conversion = np.nan
    pts, missing = _linear_score(conversion, low=0.4, high=1.3, max_points=10)
    cat.subscores.append(SubScore("Earnings-to-cash conversion", pts, 10, missing, value=conversion, unit="ratio",
                                   not_applicable=na))

    fcf_series = fcf_df["free_cash_flow"].dropna() if fcf_df is not None and not fcf_df.empty else pd.Series(dtype=float)
    if len(fcf_series) >= 2:
        fraction_positive = float((fcf_series > 0).mean())
        consistency_pts = 10.0 * fraction_positive
        missing = False
    else:
        consistency_pts, fraction_positive, missing = 5.0, np.nan, True
    cat.subscores.append(SubScore("FCF consistency (positive in every year)", consistency_pts, 10, missing,
                                   value=fraction_positive, unit="pct", not_applicable=na))

    return _redistribute_not_applicable(cat)


def _score_risk(scenario_dispersion: float, credit: dict, income_df: pd.DataFrame,
                 confidence_fraction: float, profile: CompanyProfile | None = None) -> CategoryScore:
    """Risk -- 100 points, price-independent. LOWER risk earns MORE points.

    RECALIBRATED. The audit found this category almost entirely saturated:
    the median company earned 89% of the available points on earnings
    volatility and 87% on margin volatility, so 35 of the 100 points were
    effectively granted to everyone and discriminated only against outright
    disasters. Both scales are tightened to the observed distribution, which
    restores real spread without changing what they measure.

    Leverage weight is also reduced (25 -> 18) because Net Debt/EBITDA is
    scored here for the FOURTH time -- Debt/Equity, Net Debt/EBITDA and
    Debt/EBITDA all appear in Financial Health as well.
    """
    cat = CategoryScore(name="Risk", max_points=100)
    profile = profile or CompanyProfile()
    lev_zero = profile.leverage_zero_point
    lev_applicable = not (profile.is_financial or np.isinf(lev_zero))

    pts, missing = _linear_score(scenario_dispersion, low=2.4, high=0.5, max_points=25)
    cat.subscores.append(SubScore("Valuation sensitivity (bull/bear dispersion)", pts, 25, missing,
                                   value=scenario_dispersion, unit="ratio"))

    nd_ebitda = credit.get("net_debt_to_ebitda")
    pts, missing = _linear_score(nd_ebitda, low=lev_zero if lev_applicable else 6.0, high=-0.5, max_points=18)
    cat.subscores.append(SubScore("Debt risk (Net Debt / EBITDA)", pts, 18, missing, value=nd_ebitda, unit="ratio",
                                   not_applicable=not lev_applicable))

    # Tightened from (3.0 -> 0.0) to (1.2 -> 0.05). Observed distribution:
    # median 0.32, 75th percentile 0.86, 90th 2.49. The old scale put the
    # median company at 89% and only a company mid-collapse anywhere near
    # zero, so the subscore carried almost no information.
    eps_growth = income_df["eps_growth"].dropna() if income_df is not None and not income_df.empty else pd.Series(dtype=float)
    eps_vol = eps_growth.std() if len(eps_growth) >= 2 else np.nan
    pts, missing = _linear_score(eps_vol, low=1.2, high=0.05, max_points=22)
    cat.subscores.append(SubScore("Earnings volatility", pts, 22, missing, value=eps_vol, unit="pct"))

    # Tightened from (0.20 -> 0.0) to (0.12 -> 0.005) for the same reason;
    # observed median 0.025, 75th percentile 0.072.
    op_margin_std = income_df["operating_margin"].dropna().std() if income_df is not None and not income_df.empty and len(income_df["operating_margin"].dropna()) >= 2 else np.nan
    pts, missing = _linear_score(op_margin_std, low=0.12, high=0.005, max_points=20)
    cat.subscores.append(SubScore("Margin volatility", pts, 20, missing, value=op_margin_std, unit="pct",
                                   not_applicable=profile.is_financial))

    # Data confidence: what fraction of the model's APPLICABLE inputs were
    # actual reported data rather than a neutral half-credit default. Lower
    # confidence -> fewer points, since the whole score is less trustworthy
    # when it is built on more defaults.
    #
    # Metrics that do not apply to this kind of company are excluded from
    # that fraction (see `compute_investment_score`). Previously a bank was
    # penalized twice for being a bank: once by scoring gross margin, ROIC,
    # FCF and interest coverage at a flat neutral, and again here for the
    # resulting "low data confidence" -- JPMorgan scored 0.5 out of 15.
    # Missing data still counts against confidence; inapplicable data does not.
    pts, missing = _linear_score(confidence_fraction, low=0.55, high=1.0, max_points=15)
    cat.subscores.append(SubScore("Data confidence (fraction of inputs available)", pts, 15, missing,
                                   value=confidence_fraction, unit="pct"))

    return _redistribute_not_applicable(cat)


def _price_dependent_categories(
    price: float, dcf_intrinsic_value: float, comps_low: float, comps_high: float,
    eps_ttm: float, ebitda_ttm: float, revenue_ttm: float, shares_outstanding: float, net_debt: float,
    peer_median: pd.Series, historical_pe: pd.Series, fcf_ttm: float, avg_forecast_growth: float,
    forecast_years: int, dilution_rate: float, op_margin_std: float, eps_growth_std: float,
    scenario_results, profile: CompanyProfile | None = None,
    historical_eps_growth: float = np.nan, roic_avg: float = np.nan,
) -> list[CategoryScore]:
    """Build the four PRICE-DEPENDENT categories for a given `price`. Shared
    by `compute_investment_score` (at the real current price) and
    `price_sensitivity_table` (at a range of hypothetical prices) so both
    are guaranteed to use identical logic.
    """
    profile = profile or CompanyProfile()
    target_pe, target_ev_ebitda, target_ev_revenue = _derive_price_dependent_multiples(
        price, eps_ttm, ebitda_ttm, revenue_ttm, shares_outstanding, net_debt,
    )
    peer_median_ev_ebitda = peer_median.get("EV/EBITDA") if peer_median is not None and not peer_median.empty else np.nan

    valuation = _score_valuation_and_margin_of_safety(
        price, dcf_intrinsic_value, comps_low, comps_high, target_pe, target_ev_ebitda,
        target_ev_revenue, peer_median, historical_pe, profile,
    )
    expected_return, _ = _score_expected_return(
        price, fcf_ttm, shares_outstanding, avg_forecast_growth, target_ev_ebitda,
        peer_median_ev_ebitda, forecast_years, dilution_rate, op_margin_std, eps_growth_std,
        eps_ttm=eps_ttm, profile=profile,
    )
    growth_adjusted = _score_growth_adjusted_valuation(
        target_pe, target_ev_ebitda, avg_forecast_growth,
        historical_eps_growth=historical_eps_growth, roic=roic_avg, profile=profile,
    )
    scenario_weighted, _ = _score_scenario_weighted(scenario_results, price, profile)

    return [valuation, expected_return, growth_adjusted, scenario_weighted]


def compute_investment_score(
    *,
    income_df: pd.DataFrame,
    fcf_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    credit: dict,
    roe: pd.Series,
    roic: pd.Series,
    earnings_conversion: pd.Series,
    dcf_result,
    comps_result,
    scenario_results,
    scenario_dispersion: float,
    historical_pe: pd.Series,
    dilution_rate: float,
    eps_ttm: float,
    ebitda_ttm: float,
    revenue_ttm: float,
    sector: str | None = None,
    industry: str | None = None,
) -> InvestmentScoreResult:
    """Compute the full 0-1000, price-sensitive Investment Score.

    Unlike a purely fundamentals-driven score, this pulls `price` from
    `dcf_result.current_price` and threads it through every price-dependent
    category (see module docstring) -- the whole point being that this
    number changes if the market price changes, even with every other input
    held fixed. `price_sensitivity_table()` below demonstrates exactly that.

    `sector` / `industry` select a `CompanyProfile`, which decides which
    metrics are meaningful for this kind of business and what a good value
    looks like for the ones that are. They are optional: omitted, every
    company is scored on the standard industrial profile, exactly as before.
    """
    profile = detect_company_profile(sector, industry)
    price = dcf_result.current_price
    shares_outstanding = dcf_result.shares_outstanding
    net_debt = dcf_result.net_debt
    fcf_ttm = fcf_df["free_cash_flow"].dropna().iloc[-1] if fcf_df is not None and not fcf_df.empty and not fcf_df["free_cash_flow"].dropna().empty else np.nan
    avg_forecast_growth = float(np.mean(dcf_result.assumptions.revenue_growth_path))
    forecast_years = dcf_result.assumptions.forecast_years
    op_margin_std = income_df["operating_margin"].dropna().std() if income_df is not None and not income_df.empty and len(income_df["operating_margin"].dropna()) >= 2 else np.nan
    eps_growth_std = income_df["eps_growth"].dropna().std() if income_df is not None and not income_df.empty and len(income_df["eps_growth"].dropna()) >= 2 else np.nan

    hist_eps_growth = (income_df["eps_growth"].dropna().mean()
                       if income_df is not None and not income_df.empty
                       and not income_df["eps_growth"].dropna().empty else np.nan)
    roic_avg = roic.dropna().mean() if roic is not None and not roic.dropna().empty else np.nan

    price_dependent = _price_dependent_categories(
        price, dcf_result.intrinsic_value_per_share, comps_result.comps_valuation_low, comps_result.comps_valuation_high,
        eps_ttm, ebitda_ttm, revenue_ttm, shares_outstanding, net_debt, comps_result.median_multiples,
        historical_pe, fcf_ttm, avg_forecast_growth, forecast_years, dilution_rate, op_margin_std, eps_growth_std,
        scenario_results, profile, hist_eps_growth, roic_avg,
    )
    valuation, expected_return, growth_adjusted, scenario_weighted = price_dependent

    growth = _score_growth(income_df, fcf_df, dcf_result.assumptions.revenue_growth_path[0], dilution_rate)
    profitability = _score_profitability(income_df, roe, roic, profile)
    financial_health = _score_financial_health(balance_df, credit, profile)
    cash_flow = _score_cash_flow_quality(fcf_df, earnings_conversion, profile)

    business_quality_categories = [growth, profitability, financial_health, cash_flow]

    # Data confidence: the fraction of all APPLICABLE subscores computed so
    # far (across both price-dependent and business-quality categories) that
    # were backed by real data rather than a neutral default.
    #
    # Subscores marked not-applicable are excluded from both the numerator
    # and denominator: a bank having no gross margin is not a gap in the data
    # feed, and counting it as one penalized every financial company a second
    # time for a fact already handled by redistribution.
    provisional_categories = price_dependent + business_quality_categories
    all_subscores = [s for c in provisional_categories for s in c.subscores if not s.not_applicable]
    confidence_fraction = 1.0 - (sum(1 for s in all_subscores if s.missing) / len(all_subscores)) if all_subscores else 0.5

    risk = _score_risk(scenario_dispersion, credit, income_df, confidence_fraction, profile)

    categories = price_dependent + business_quality_categories + [risk]
    total = sum(c.points for c in categories)
    total = float(min(max(total, 0), MAX_SCORE))
    rating = get_rating(total)

    positive_factors, negative_factors = _generate_factors(categories)

    business_quality_score = sum(c.points for c in business_quality_categories) / sum(
        c.max_points for c in business_quality_categories) * 100
    valuation_attractiveness_score = (valuation.points + growth_adjusted.points) / (
        valuation.max_points + growth_adjusted.max_points) * 100
    expected_return_score = (expected_return.points + scenario_weighted.points) / (
        expected_return.max_points + scenario_weighted.max_points) * 100
    risk_score = risk.points / risk.max_points * 100

    return InvestmentScoreResult(
        total=total,
        rating=rating,
        categories=categories,
        positive_factors=positive_factors,
        negative_factors=negative_factors,
        business_quality_score=business_quality_score,
        valuation_attractiveness_score=valuation_attractiveness_score,
        expected_return_score=expected_return_score,
        risk_score=risk_score,
    )


def price_sensitivity_table(
    *,
    income_df: pd.DataFrame,
    fcf_df: pd.DataFrame,
    dcf_result,
    comps_result,
    scenario_results,
    historical_pe: pd.Series,
    dilution_rate: float,
    eps_ttm: float,
    ebitda_ttm: float,
    revenue_ttm: float,
    business_quality_points: float,
    risk_points: float,
    price_multipliers: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
    sector: str | None = None,
    industry: str | None = None,
    roe: pd.Series | None = None,
    roic: pd.Series | None = None,
) -> pd.DataFrame:
    """Recompute the total Investment Score at a range of hypothetical
    prices, holding every non-price input fixed, to make explicit that the
    score represents the attractiveness of the investment AT A PRICE
    (see project spec: "Price Sensitivity Test").

    Business Quality (Growth/Profitability/Financial Health/Cash Flow) and
    Risk are price-independent and are added back in unchanged at every
    hypothetical price -- only the four price-dependent categories are
    recomputed, using `_price_dependent_categories()`, the same function
    `compute_investment_score()` uses for the real current price. This
    guarantees the two never drift out of sync and that the relationship
    between price and score emerges from the underlying formulas rather
    than being hard-coded.
    """
    price = dcf_result.current_price
    shares_outstanding = dcf_result.shares_outstanding
    net_debt = dcf_result.net_debt
    fcf_ttm = fcf_df["free_cash_flow"].dropna().iloc[-1] if fcf_df is not None and not fcf_df.empty and not fcf_df["free_cash_flow"].dropna().empty else np.nan
    avg_forecast_growth = float(np.mean(dcf_result.assumptions.revenue_growth_path))
    forecast_years = dcf_result.assumptions.forecast_years
    op_margin_std = income_df["operating_margin"].dropna().std() if income_df is not None and not income_df.empty and len(income_df["operating_margin"].dropna()) >= 2 else np.nan
    eps_growth_std = income_df["eps_growth"].dropna().std() if income_df is not None and not income_df.empty and len(income_df["eps_growth"].dropna()) >= 2 else np.nan

    # The same company profile and quality inputs the real score used, so the
    # sensitivity table stays consistent with the headline number.
    profile = detect_company_profile(sector, industry)
    hist_eps_growth = (income_df["eps_growth"].dropna().mean()
                       if income_df is not None and not income_df.empty
                       and not income_df["eps_growth"].dropna().empty else np.nan)
    roic_avg = roic.dropna().mean() if roic is not None and not roic.dropna().empty else np.nan

    rows = []
    for mult in price_multipliers:
        hypothetical_price = price * mult
        cats = _price_dependent_categories(
            hypothetical_price, dcf_result.intrinsic_value_per_share, comps_result.comps_valuation_low,
            comps_result.comps_valuation_high, eps_ttm, ebitda_ttm, revenue_ttm, shares_outstanding, net_debt,
            comps_result.median_multiples, historical_pe, fcf_ttm, avg_forecast_growth, forecast_years,
            dilution_rate, op_margin_std, eps_growth_std, scenario_results, profile, hist_eps_growth, roic_avg,
        )
        price_dependent_points = sum(c.points for c in cats)
        total = float(min(max(price_dependent_points + business_quality_points + risk_points, 0), MAX_SCORE))
        rows.append({
            "Price": hypothetical_price,
            "Change": f"{(mult - 1) * 100:+.0f}%",
            "Score": round(total),
            "Rating": get_rating(total),
        })

    return pd.DataFrame(rows)


def _generate_factors(categories: list[CategoryScore]) -> tuple[list[str], list[str]]:
    """Generate human-readable positive/negative factors strictly from
    subscore fractions -- a subscore earning >=75% of its available points
    is a positive factor, <=30% is a negative factor. Nothing here is
    hand-written per company; it is entirely derived from the computed
    scores above.
    """
    positives, negatives = [], []
    for cat in categories:
        for sub in cat.subscores:
            if sub.missing or sub.max_points == 0:
                continue
            frac = sub.points / sub.max_points
            if frac >= 0.75:
                positives.append(f"{sub.label} ({cat.name})")
            elif frac <= 0.30:
                negatives.append(f"{sub.label} ({cat.name})")

    return positives[:6], negatives[:6]
