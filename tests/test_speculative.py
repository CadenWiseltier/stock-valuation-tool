"""Unit tests for the redesigned 0-1000 Speculation Score.

The score answers "how much potential does this stock have to EXPLODE in
value if its bullish thesis plays out?" -- so these tests concentrate on the
properties that definition demands: that opportunity size drives the score,
that a trendy keyword alone cannot, that a high valuation is NOT penalized,
and that volatility plays no part at all.
"""

import numpy as np
import pandas as pd
import pytest

from src.dcf import build_baseline_assumptions, calculate_wacc, run_dcf, _cagr
from src.financial_analysis import (
    credit_metrics, diluted_shares_growth_rate, fcf_summary, income_statement_summary,
)
from src.scenarios import SCENARIO_ADJUSTMENTS, run_scenarios
from src.scoring import CategoryScore, SubScore
from src.speculative import (
    EXPLANATION_REQUIRED_ABOVE, EXPLOSIVE_QUALIFICATION_ABOVE, MAX_SPECULATIVE_SCORE,
    SPECULATIVE_RATING_THRESHOLDS, _CATEGORY_RISK_TEMPLATES, _CATEGORY_THESIS_TEMPLATES,
    _generate_speculative_narrative, _log_score, compute_speculative_score,
    get_speculative_rating, potential_market_cap, probability_weighted_scenario_value,
    theme_evidence_strength,
)
from src.themes import MATURE_INDUSTRY, THEMES, classify_theme


@pytest.fixture
def speculative_inputs(synthetic_bundle):
    income_df = income_statement_summary(synthetic_bundle)
    fcf_df = fcf_summary(synthetic_bundle)
    credit = credit_metrics(synthetic_bundle)
    dilution_rate = diluted_shares_growth_rate(synthetic_bundle)

    assumptions = build_baseline_assumptions(synthetic_bundle)
    wacc = calculate_wacc(synthetic_bundle)
    dcf_result = run_dcf(synthetic_bundle, assumptions, wacc)
    scenario_results = run_scenarios(synthetic_bundle, assumptions, wacc)

    return dict(
        bundle=synthetic_bundle, income_df=income_df, fcf_df=fcf_df, credit=credit,
        dcf_result=dcf_result, scenario_results=scenario_results, dilution_rate=dilution_rate,
    )


# ---------------------------------------------------------------------------
# Structure and bounds
# ---------------------------------------------------------------------------
def test_category_weights_sum_to_1000(speculative_inputs):
    result = compute_speculative_score(**speculative_inputs)
    assert sum(c.max_points for c in result.categories) == MAX_SPECULATIVE_SCORE
    assert [c.max_points for c in result.categories] == [150, 175, 150, 125, 150, 100, 75, 75]


def test_score_and_subscores_within_bounds(speculative_inputs):
    result = compute_speculative_score(**speculative_inputs)
    assert 0 <= result.total <= 1000
    assert 0 <= result.confidence <= 100
    for cat in result.categories:
        assert 0 <= cat.points <= cat.max_points
        for sub in cat.subscores:
            assert 0 <= sub.points <= sub.max_points


def test_rating_bands_match_the_specified_scale():
    """The bands are the user-facing meaning of the score, so they are
    pinned exactly: 0-150 minimal, 151-300 low, 301-450 moderate, 451-600
    high, 601-750 very high, 751-850 extreme, 851-950 exceptional/explosive,
    951-1000 once-in-a-generation."""
    assert get_speculative_rating(0) == "Minimal Speculative Upside"
    assert get_speculative_rating(150) == "Minimal Speculative Upside"
    assert get_speculative_rating(151) == "Low Speculation"
    assert get_speculative_rating(300) == "Low Speculation"
    assert get_speculative_rating(301) == "Moderate Speculation"
    assert get_speculative_rating(450) == "Moderate Speculation"
    assert get_speculative_rating(451) == "High Speculation"
    assert get_speculative_rating(600) == "High Speculation"
    assert get_speculative_rating(601) == "Very High Speculation"
    assert get_speculative_rating(750) == "Very High Speculation"
    assert get_speculative_rating(751) == "Extreme Speculation"
    assert get_speculative_rating(850) == "Extreme Speculation"
    assert get_speculative_rating(851) == "Exceptional / Explosive Speculation"
    assert get_speculative_rating(950) == "Exceptional / Explosive Speculation"
    assert get_speculative_rating(951) == "Once-in-a-Generation Speculative Opportunity"
    assert get_speculative_rating(1000) == "Once-in-a-Generation Speculative Opportunity"
    assert len(SPECULATIVE_RATING_THRESHOLDS) == 8


# ---------------------------------------------------------------------------
# The defining behaviours of the redesign
# ---------------------------------------------------------------------------
def test_volatility_and_beta_play_no_part_in_the_score(speculative_inputs):
    """Explicit requirement: speculation is not volatility. Beta, scenario
    dispersion, short interest and popularity must not appear anywhere. A
    company's score must be completely unchanged by its beta."""
    baseline = compute_speculative_score(**speculative_inputs)

    volatile = dict(speculative_inputs)
    volatile_bundle = _clone_bundle(volatile["bundle"], {"beta": 4.5})
    volatile["bundle"] = volatile_bundle
    result = compute_speculative_score(**volatile)

    assert result.total == pytest.approx(baseline.total)

    labels = " ".join(s.label for c in result.categories for s in c.subscores).lower()
    for banned in ("beta", "volatil", "dispersion", "short interest"):
        assert banned not in labels


def test_a_high_valuation_multiple_is_not_penalized(speculative_inputs):
    """A rich P/E or EV/Sales belongs to the Fundamental Score. The only way
    price may enter the Speculation Score is through market-cap SCALE, so
    changing the reported trailing P/E must not move the score at all."""
    baseline = compute_speculative_score(**speculative_inputs)

    expensive = dict(speculative_inputs)
    expensive["bundle"] = _clone_bundle(
        speculative_inputs["bundle"],
        {"trailingPE": 480.0, "forwardPE": 300.0, "priceToSalesTrailing12Months": 95.0,
         "enterpriseToRevenue": 95.0, "enterpriseToEbitda": 220.0},
    )
    assert compute_speculative_score(**expensive).total == pytest.approx(baseline.total)


def test_smaller_market_cap_raises_asymmetric_upside(speculative_inputs):
    """The user's own framing: $5B -> $50B is 10x and must score far higher
    than $50B -> $100B at 2x. With everything else held constant, a smaller
    company therefore has more room to multiply."""
    big = dict(speculative_inputs)
    big["bundle"] = _clone_bundle(speculative_inputs["bundle"], {"marketCap": 400e9})
    small = dict(speculative_inputs)
    small["bundle"] = _clone_bundle(speculative_inputs["bundle"], {"marketCap": 2e9})

    big_result = compute_speculative_score(**big)
    small_result = compute_speculative_score(**small)

    assert small_result.upside_multiple > big_result.upside_multiple
    big_upside = next(c for c in big_result.categories if c.name == "5-10 Year Asymmetric Upside")
    small_upside = next(c for c in small_result.categories if c.name == "5-10 Year Asymmetric Upside")
    assert small_upside.points > big_upside.points


def test_emerging_industry_scores_higher_than_mature_industry(speculative_inputs):
    """The core claim of the redesign: identical financials in an emerging,
    large-TAM industry must produce a materially higher score than in a
    mature one. This is tested by changing ONLY the business description
    used for industry classification."""
    mature = dict(speculative_inputs)
    mature["bundle"] = _clone_bundle(
        speculative_inputs["bundle"],
        {"longBusinessSummary": "The company manufactures and distributes packaged beverages "
                                "and snack foods to grocery retailers."},
    )
    emerging = dict(speculative_inputs)
    emerging["bundle"] = _clone_bundle(
        speculative_inputs["bundle"],
        {"longBusinessSummary": "The company develops trapped ion quantum computing systems "
                                "and quantum computer hardware for enterprise customers."},
    )

    mature_result = compute_speculative_score(**mature)
    emerging_result = compute_speculative_score(**emerging)

    assert mature_result.theme.key == MATURE_INDUSTRY.key
    assert emerging_result.theme.key == "quantum_computing"
    assert emerging_result.total > mature_result.total + 100


def test_keyword_alone_cannot_manufacture_a_high_score(synthetic_bundle):
    """Explicit requirement: "Do NOT automatically give a high score simply
    because a company uses a trendy keyword." A shell with a quantum
    business description but no revenue, no growth, no R&D, no margin and no
    capital investment must be gated down to a fraction of the industry's
    raw optionality."""
    quantum_shell = _clone_bundle(
        synthetic_bundle,
        {"longBusinessSummary": "A quantum computing company developing qubit technology."},
    )
    empty_income = pd.DataFrame()
    empty_fcf = pd.DataFrame()

    weak_evidence = theme_evidence_strength(quantum_shell, empty_income, empty_fcf)
    strong_evidence = theme_evidence_strength(
        quantum_shell,
        pd.DataFrame({"revenue": [8e8, 1.4e9], "gross_margin": [0.62, 0.70]},
                     index=pd.to_datetime(["2023-12-31", "2024-12-31"])),
        pd.DataFrame({"capex": [2e8, 3e8]}, index=pd.to_datetime(["2023-12-31", "2024-12-31"])),
    )
    assert weak_evidence < strong_evidence

    from src.speculative import _score_optionality
    from src.themes import classify_theme as _classify
    theme = _classify("ZZZZ", quantum_shell.info).theme
    assert theme.key == "quantum_computing"

    gated = _score_optionality(theme, evidence=0.0)
    ungated = _score_optionality(theme, evidence=1.0)
    assert gated.points < ungated.points * 0.5


def test_missing_data_is_neutral_never_zero(speculative_inputs):
    """Project-wide rule: an unavailable metric scores neutral half-credit
    and is flagged missing, so the score reflects the opportunity rather
    than the completeness of the data feed."""
    result = compute_speculative_score(**speculative_inputs)
    for cat in result.categories:
        for sub in cat.subscores:
            if sub.missing:
                assert sub.points == pytest.approx(sub.max_points * 0.5)
                assert np.isnan(sub.value)


def test_genuine_zero_and_negative_values_are_preserved(speculative_inputs):
    """A real zero or negative number is data, not a data failure: negative
    dilution (buybacks) must display and score as the true negative value."""
    inputs = dict(speculative_inputs)
    inputs["dilution_rate"] = -0.04
    result = compute_speculative_score(**inputs)
    win = next(c for c in result.categories if c.name == "Probability of Becoming a Major Winner")
    dilution_sub = next(s for s in win.subscores if "Dilution discipline" in s.label)
    assert dilution_sub.value == pytest.approx(-0.04)
    assert not dilution_sub.missing
    assert dilution_sub.points == pytest.approx(dilution_sub.max_points)


# ---------------------------------------------------------------------------
# The opportunity model
# ---------------------------------------------------------------------------
def test_potential_market_cap_scales_with_evidence():
    """Two companies in the same industry must not share one identical
    upside estimate -- the positioning factor derived from observable
    evidence is what separates them."""
    theme = next(t for t in THEMES if t.key == "quantum_computing")
    weak, weak_pos = potential_market_cap(theme, evidence=0.1, revenue=10e6)
    strong, strong_pos = potential_market_cap(theme, evidence=0.9, revenue=10e6)
    assert strong > weak
    assert 0.25 <= weak_pos < strong_pos <= 2.0


def test_potential_revenue_is_floored_at_current_revenue():
    """A successful-case scenario must never imply the company shrinks."""
    theme = MATURE_INDUSTRY
    huge_revenue = 400e9
    potential, _ = potential_market_cap(theme, evidence=0.5, revenue=huge_revenue)
    assert potential >= huge_revenue * theme.winner_ev_sales


def test_log_score_treats_nonpositive_as_real_data_not_missing():
    """An upside multiple below zero is a genuine (bad) result, not a
    missing input -- it must score 0 without being flagged missing."""
    pts, missing = _log_score(-2.0, low=1.5, high=15.0, max_points=100)
    assert pts == 0.0
    assert missing is False

    pts, missing = _log_score(np.nan, low=1.5, high=15.0, max_points=100)
    assert pts == pytest.approx(50.0)
    assert missing is True


def test_log_score_inverted_scale_handles_nonpositive_at_the_top():
    """On a descending scale (smaller is better, e.g. revenue base), a
    non-positive value sits at the favourable end, not the unfavourable one."""
    pts, missing = _log_score(0.0, low=50e9, high=50e6, max_points=40)
    assert pts == pytest.approx(40.0)
    assert missing is False


# ---------------------------------------------------------------------------
# Theme classification
# ---------------------------------------------------------------------------
def test_theme_tables_are_internally_consistent():
    keys = [t.key for t in THEMES]
    assert len(keys) == len(set(keys)), "duplicate theme keys"
    for t in list(THEMES) + [MATURE_INDUSTRY]:
        assert t.tam_usd > 0
        assert 0.0 <= t.commercial_maturity <= 1.0
        assert 0.0 <= t.breakthrough_potential <= 1.0
        assert 0.0 <= t.catalyst_intensity <= 1.0
        assert 0.0 <= t.winner_take_most <= 1.0
        assert 0.0 < t.plausible_winner_share <= 1.0
        assert t.winner_ev_sales > 0
        assert t.source_note, f"{t.key} must document where its TAM estimate came from"


def test_explicit_ticker_mapping_beats_keywords():
    match = classify_theme("IONQ", {"longBusinessSummary": "Some generic description."})
    assert match.theme.key == "quantum_computing"
    assert match.match_basis == "ticker"


def test_unrecognized_company_falls_back_to_mature_industry():
    match = classify_theme("ZZZZ", {"longBusinessSummary": "Operates regional dry cleaning stores."})
    assert match.theme.key == MATURE_INDUSTRY.key
    assert match.match_basis == "none"
    assert match.is_emerging is False


def test_classification_is_deterministic_across_multiple_matches():
    """A company matching several themes must resolve to the same primary
    theme every time, chosen by an explicit ranking rather than table order."""
    info = {"longBusinessSummary": "Operates a data center fleet for artificial intelligence and "
                                   "high performance computing workloads, and also performs "
                                   "bitcoin mining and other cryptocurrency mining operations."}
    first = classify_theme("XXXX", info)
    assert all(classify_theme("XXXX", info).theme.key == first.theme.key for _ in range(5))
    assert first.secondary, "other matched themes should be retained for display"


def test_incidental_keyword_mention_does_not_reclassify_a_company():
    """Regression test for a real false positive found in testing: Exxon
    Mobil's business summary lists "low-carbon data center" among a dozen
    other business lines, which reclassified an integrated oil major as an
    AI-infrastructure company. One passing mention buried in a long summary
    must never be enough."""
    xom_like = {
        "sector": "Energy",
        "industry": "Oil & Gas Integrated",
        "longName": "Exxon Mobil Corporation",
        "longBusinessSummary": (
            "The company explores for and produces crude oil and natural gas. It also offers "
            "carbon capture and storage, hydrogen, lower-emission fuels, proxxima resin systems, "
            "carbon materials, low-carbon data center, and lithium. In addition, the company "
            "offers aviation fuel."
        ),
    }
    match = classify_theme("XOM", xom_like)
    assert match.theme.key == MATURE_INDUSTRY.key
    assert match.match_basis == "none"


def test_a_primary_field_match_is_sufficient_on_its_own():
    """The flip side of the threshold: when the company's own industry
    classification names the technology, one match IS decisive -- that field
    states what the company is, rather than mentioning something it does."""
    match = classify_theme("ZZZZ", {"industry": "Semiconductors",
                                    "longBusinessSummary": "Designs and sells chips."})
    assert match.theme.key == "advanced_semis"
    assert match.match_basis == "keyword"


# ---------------------------------------------------------------------------
# Explanation and narrative
# ---------------------------------------------------------------------------
def test_explanation_is_absent_below_the_threshold_and_present_above(speculative_inputs):
    result = compute_speculative_score(**speculative_inputs)
    if result.total >= EXPLANATION_REQUIRED_ABOVE:
        assert result.explanation is not None
    else:
        assert result.explanation is None


def test_explanation_contains_all_five_required_parts():
    """Requirement: above 700 the tool must explain (1) why it is explosive,
    (2) the industry/TAM, (3) catalysts, (4) the upside scenario, (5) the
    biggest reason it could fail."""
    from src.speculative import _build_explanation
    theme = next(t for t in THEMES if t.key == "quantum_computing")
    match = classify_theme("IONQ", {})
    categories = [CategoryScore(name=n, max_points=m, subscores=[SubScore("x", m * 0.8, m)])
                  for n, m in (("TAM / Future Market Opportunity", 150),
                               ("Explosive Growth Potential", 175),
                               ("5-10 Year Asymmetric Upside", 150))]
    exp = _build_explanation(
        total=880, theme=theme, match=match, categories=categories, revenue=43e6,
        mcap=15e9, potential_mcap=300e9, upside_multiple=20.0, evidence=0.8,
        company_name="Test Co",
    )
    assert exp is not None
    assert exp.why_explosive and exp.industry_and_tam and exp.upside_scenario and exp.biggest_risk
    assert len(exp.catalysts) >= 1
    # Above 850 the explosive qualification is mandatory.
    assert exp.explosive_qualification
    assert "EXPLOSIVE" in exp.explosive_qualification


def test_explosive_qualification_only_above_850():
    from src.speculative import _build_explanation
    theme = next(t for t in THEMES if t.key == "quantum_computing")
    match = classify_theme("IONQ", {})
    categories = [CategoryScore(name="TAM / Future Market Opportunity", max_points=150,
                                subscores=[SubScore("x", 120.0, 150.0)])]
    kwargs = dict(theme=theme, match=match, categories=categories, revenue=43e6, mcap=15e9,
                  potential_mcap=300e9, upside_multiple=20.0, evidence=0.8, company_name="Test Co")

    assert _build_explanation(total=EXPLANATION_REQUIRED_ABOVE - 1, **kwargs) is None
    mid = _build_explanation(total=EXPLOSIVE_QUALIFICATION_ABOVE - 1, **kwargs)
    assert mid is not None and mid.explosive_qualification == ""
    high = _build_explanation(total=EXPLOSIVE_QUALIFICATION_ABOVE, **kwargs)
    assert high is not None and high.explosive_qualification != ""


def test_narrative_bullets_are_never_fabricated(speculative_inputs):
    """Every thesis/risk bullet must come verbatim from the fixed template
    dictionaries -- never an invented product, deal, or event."""
    result = compute_speculative_score(**speculative_inputs)
    allowed = set(_CATEGORY_THESIS_TEMPLATES.values()) | set(_CATEGORY_RISK_TEMPLATES.values())
    for bullet in result.thesis + result.risks:
        assert bullet in allowed


def test_narrative_templates_cover_every_category(speculative_inputs):
    result = compute_speculative_score(**speculative_inputs)
    for cat in result.categories:
        assert cat.name in _CATEGORY_THESIS_TEMPLATES
        assert cat.name in _CATEGORY_RISK_TEMPLATES


def test_generate_narrative_respects_thresholds():
    strong = CategoryScore(name="TAM / Future Market Opportunity", max_points=150,
                           subscores=[SubScore("x", 145.0, 150.0)])
    weak = CategoryScore(name="Explosive Growth Potential", max_points=175,
                         subscores=[SubScore("y", 10.0, 175.0)])
    middling = CategoryScore(name="Catalyst Potential", max_points=125,
                             subscores=[SubScore("z", 62.5, 125.0)])
    thesis, risks = _generate_speculative_narrative([strong, weak, middling])
    assert _CATEGORY_THESIS_TEMPLATES["TAM / Future Market Opportunity"] in thesis
    assert _CATEGORY_RISK_TEMPLATES["Explosive Growth Potential"] in risks
    assert _CATEGORY_THESIS_TEMPLATES["Catalyst Potential"] not in thesis
    assert _CATEGORY_RISK_TEMPLATES["Catalyst Potential"] not in risks


# ---------------------------------------------------------------------------
# Regression guards on shared machinery
# ---------------------------------------------------------------------------
def test_extreme_bull_tier_still_exists_for_the_fundamental_score():
    assert set(SCENARIO_ADJUSTMENTS.keys()) == {"bear", "base", "bull", "extreme_bull"}


def test_probability_weighted_value_uses_all_four_tiers(speculative_inputs):
    value = probability_weighted_scenario_value(speculative_inputs["scenario_results"])
    results = speculative_inputs["scenario_results"]
    assert not np.isnan(value)
    assert results["bear"].dcf.intrinsic_value_per_share < value < results["extreme_bull"].dcf.intrinsic_value_per_share


def test_probability_weighted_value_is_missing_not_zero_without_extreme_bull():
    class _FakeDCF:
        def __init__(self, v):
            self.intrinsic_value_per_share = v

    class _FakeResult:
        def __init__(self, v):
            self.dcf = _FakeDCF(v)

    incomplete = {"bear": _FakeResult(50.0), "base": _FakeResult(70.0), "bull": _FakeResult(100.0)}
    assert np.isnan(probability_weighted_scenario_value(incomplete))


def test_ebit_sign_flip_does_not_crash_cagr():
    """Regression test for a real bug: a CAGR across a series that flips from
    positive to negative must return NaN cleanly rather than raising from
    a negative number to a fractional power."""
    with np.errstate(invalid="raise"):
        assert np.isnan(_cagr(pd.Series([100.0, 50.0, -20.0])))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clone_bundle(bundle, info_overrides: dict):
    """Return a shallow copy of `bundle` with `info` fields overridden, so a
    test can vary one input while holding all statements constant."""
    import copy
    clone = copy.copy(bundle)
    clone.info = {**bundle.info, **info_overrides}
    return clone
