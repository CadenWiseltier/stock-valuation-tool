"""Unit tests for src/scoring.py: the 0-1000, price-sensitive Investment
Score."""

import numpy as np
import pandas as pd
import pytest

from src.comparables import ComparablesResult
from src.dcf import build_baseline_assumptions, calculate_wacc, run_dcf
from src.financial_analysis import (
    balance_sheet_summary, credit_metrics, earnings_to_cash_conversion,
    fcf_summary, income_statement_summary, roe_series, roic_series,
)
from src.scenarios import run_scenarios, scenario_dispersion
from src.scoring import (
    MAX_SCORE, compute_investment_score, get_rating, price_sensitivity_table, _linear_score,
    DEEP_DISCOUNT_UPSIDE_FLOOR, DEEP_DOWNSIDE_FLOOR, _score_scenario_weighted, _score_expected_return,
    MAX_CREDIBLE_DCF_UPSIDE, CategoryScore, CompanyProfile, SubScore,
    _redistribute_not_applicable, _sanitize_upside, _score_growth_adjusted_valuation,
    _historical_valuation_percentile_score, detect_company_profile,
)


class _FakeDCF:
    def __init__(self, intrinsic_value_per_share):
        self.intrinsic_value_per_share = intrinsic_value_per_share


class _FakeScenarioResult:
    def __init__(self, intrinsic_value_per_share):
        self.dcf = _FakeDCF(intrinsic_value_per_share)


def _synthetic_comps_result() -> ComparablesResult:
    """A hand-built comps result so scoring tests do not depend on network
    access to fetch real peer data."""
    peer_table = pd.DataFrame({
        "Ticker": ["AAA", "BBB", "CCC"],
        "Company": ["Peer A", "Peer B", "Peer C"],
        "P/E": [25.0, 28.0, 22.0],
        "EV/EBITDA": [15.0, 16.0, 14.0],
        "EV/Revenue": [5.0, 5.5, 4.5],
        "P/S": [4.0, 4.5, 3.5],
    })
    median_multiples = peer_table[["P/E", "EV/EBITDA", "EV/Revenue", "P/S"]].median()
    mean_multiples = peer_table[["P/E", "EV/EBITDA", "EV/Revenue", "P/S"]].mean()
    return ComparablesResult(
        peer_table=peer_table,
        mean_multiples=mean_multiples,
        median_multiples=median_multiples,
        implied_values={"P/E": 105.0, "EV/EBITDA": 110.0, "EV/Revenue": 95.0},
        comps_valuation_low=95.0,
        comps_valuation_high=110.0,
        peer_tickers=["AAA", "BBB", "CCC"],
        sector="Technology",
        peer_sector="Technology",
    )


@pytest.fixture
def full_score_inputs(synthetic_bundle):
    income_df = income_statement_summary(synthetic_bundle)
    fcf_df = fcf_summary(synthetic_bundle)
    balance_df = balance_sheet_summary(synthetic_bundle)
    credit = credit_metrics(synthetic_bundle)
    roe = roe_series(synthetic_bundle)
    roic = roic_series(synthetic_bundle)
    earnings_conversion = earnings_to_cash_conversion(synthetic_bundle)

    assumptions = build_baseline_assumptions(synthetic_bundle)
    wacc = calculate_wacc(synthetic_bundle)
    dcf_result = run_dcf(synthetic_bundle, assumptions, wacc)

    comps_result = _synthetic_comps_result()
    scenario_results = run_scenarios(synthetic_bundle, assumptions, wacc)
    dispersion = scenario_dispersion(scenario_results)

    return dict(
        income_df=income_df, fcf_df=fcf_df, balance_df=balance_df, credit=credit,
        roe=roe, roic=roic, earnings_conversion=earnings_conversion,
        dcf_result=dcf_result, comps_result=comps_result, scenario_results=scenario_results,
        scenario_dispersion=dispersion, historical_pe=pd.Series(dtype=float), dilution_rate=0.0,
        eps_ttm=1.815, ebitda_ttm=314.6, revenue_ttm=1210.0,
    )


def test_category_max_points_sum_to_1000():
    """Structural guard: 250 + 200 + 100 + 100 + 70 + 70 + 70 + 40 + 100
    must equal exactly 1000, per the price-sensitive scoring design."""
    weights = [250, 200, 100, 100, 70, 70, 70, 40, 100]
    assert sum(weights) == 1000
    assert MAX_SCORE == 1000


def test_investment_score_within_bounds(full_score_inputs):
    result = compute_investment_score(**full_score_inputs)
    assert 0 <= result.total <= 1000
    assert sum(c.max_points for c in result.categories) == 1000


def test_investment_score_category_points_never_exceed_max(full_score_inputs):
    result = compute_investment_score(**full_score_inputs)
    for cat in result.categories:
        assert 0 <= cat.points <= cat.max_points
        for sub in cat.subscores:
            assert 0 <= sub.points <= sub.max_points


def test_headline_scores_within_0_100(full_score_inputs):
    result = compute_investment_score(**full_score_inputs)
    assert 0 <= result.business_quality_score <= 100
    assert 0 <= result.valuation_attractiveness_score <= 100
    assert 0 <= result.expected_return_score <= 100
    assert 0 <= result.risk_score <= 100


def test_deeply_negative_upside_still_earns_partial_credit_not_zero():
    """Regression test for a real bug report: with the original -20%/-30%
    scoring floors, every richly-valued real stock tested (a conservative
    5-year FCFF DCF routinely implies -60% to -130% "upside" for names
    like AAPL/AMZN/NVDA/TSLA) clamped to exactly 0 on the DCF-based and
    scenario-weighted subscores, destroying any differentiation between a
    merely expensive stock and an absurdly expensive one. The floor was
    widened to -100% (DEEP_DISCOUNT_UPSIDE_FLOOR) / -130%
    (DEEP_DOWNSIDE_FLOOR) specifically so a "moderately overvalued" input
    still earns partial credit rather than being indistinguishable from a
    "wildly overvalued" one.
    """
    # A value between the old (-0.30) and new (-1.00) floor must now score
    # strictly above zero, where it used to be clamped to exactly zero.
    pts, missing = _linear_score(-0.65, low=DEEP_DISCOUNT_UPSIDE_FLOOR, high=0.40, max_points=70)
    assert pts > 0
    assert not missing

    pts, missing = _linear_score(-0.90, low=DEEP_DOWNSIDE_FLOOR, high=0.0, max_points=25)
    assert pts > 0


def test_asymmetry_uses_value_differences_not_price_relative_ratio():
    """Regression test: when a stock is deeply overvalued even in the Bull
    case (all three scenario values below current price), the OLD
    "Bull upside / Bear upside" ratio divided two negative numbers and
    produced a nonsensical/sign-flipped result, always clamping the
    asymmetry subscore to zero regardless of how favorable the actual
    scenario skew was. The new (Bull - Base) / (Base - Bear) formulation
    stays meaningful in that situation.
    """
    scenario_results = {
        "bear": _FakeScenarioResult(50.0),
        "base": _FakeScenarioResult(70.0),
        "bull": _FakeScenarioResult(100.0),  # all three below price=200
    }
    cat, _ = _score_scenario_weighted(scenario_results, price=200.0)
    asymmetry_sub = next(s for s in cat.subscores if "asymmetry" in s.label.lower())
    # (100-70)/(70-50) = 1.5 -> a favorable, well-defined ratio, not zero.
    assert asymmetry_sub.points > 0
    assert not asymmetry_sub.missing


def test_asymmetry_is_missing_not_zero_when_scenario_ordering_breaks_down():
    """Regression test for a real edge case found in testing (a company
    whose thin-to-negative margins make growth value-destructive): when
    the Bull case is not actually better than Base (or Base not better
    than Bear), the ratio is not meaningful and must be reported as
    missing/neutral rather than a misleadingly bad score of zero.
    """
    scenario_results = {
        "bear": _FakeScenarioResult(-18.0),
        "base": _FakeScenarioResult(-21.0),  # worse than bear
        "bull": _FakeScenarioResult(-23.0),  # worse than base
    }
    cat, _ = _score_scenario_weighted(scenario_results, price=100.0)
    asymmetry_sub = next(s for s in cat.subscores if "asymmetry" in s.label.lower())
    assert asymmetry_sub.missing is True
    assert asymmetry_sub.points == pytest.approx(asymmetry_sub.max_points * 0.5)


def test_rating_thresholds():
    """Pinned to the project's stated calibration bands: 900+ exceptional,
    800-899 very strong, 700-799 strong/attractive, 600-699 moderately
    attractive, 500-599 fair/neutral, 400-499 below average, under 400 weak.
    The seven label values are unchanged; only the boundaries moved."""
    assert get_rating(1000) == "Exceptional"
    assert get_rating(900) == "Exceptional"
    assert get_rating(899) == "Strong Buy"
    assert get_rating(800) == "Strong Buy"
    assert get_rating(799) == "Buy"
    assert get_rating(700) == "Buy"
    assert get_rating(699) == "Moderate / Watch"
    assert get_rating(600) == "Moderate / Watch"
    assert get_rating(599) == "Hold"
    assert get_rating(500) == "Hold"
    assert get_rating(499) == "Weak"
    assert get_rating(400) == "Weak"
    assert get_rating(399) == "Avoid"
    assert get_rating(0) == "Avoid"


def test_linear_score_clamps_and_handles_missing():
    assert _linear_score(0.0, low=0.0, high=0.30, max_points=30)[0] == pytest.approx(0.0)
    assert _linear_score(0.30, low=0.0, high=0.30, max_points=30)[0] == pytest.approx(30.0)
    assert _linear_score(1.0, low=0.0, high=0.30, max_points=30)[0] == pytest.approx(30.0)
    assert _linear_score(-1.0, low=0.0, high=0.30, max_points=30)[0] == pytest.approx(0.0)
    pts, _ = _linear_score(0.0, low=2.5, high=0.0, max_points=30)
    assert pts == pytest.approx(30.0)
    pts, missing = _linear_score(float("nan"), low=0.0, high=1.0, max_points=40)
    assert pts == pytest.approx(20.0)
    assert missing is True


def test_quality_alone_cannot_reach_strong_buy(full_score_inputs):
    """Core design requirement from the user's price-sensitivity spec:
    Business Quality (Growth + Profitability + Financial Health + Cash
    Flow Quality) is capped at 250 of 1000 points, so even a company that
    scores a PERFECT 250/250 on quality cannot reach "Strong Buy" (750)
    without ALSO scoring at least 500 of the remaining 750 price-dependent
    + risk points. This directly encodes "a great company at a terrible
    price should not necessarily be a Buy."
    """
    result = compute_investment_score(**full_score_inputs)
    business_quality_categories = [c for c in result.categories
                                    if c.name in ("Growth", "Profitability", "Financial Health", "Cash Flow Quality")]
    max_quality_points = sum(c.max_points for c in business_quality_categories)
    assert max_quality_points == 250
    assert max_quality_points < 750  # below the Strong Buy threshold on its own


def test_price_sensitivity_score_decreases_as_price_increases(full_score_inputs):
    """The whole point of the redesign: paying more for the SAME company
    (same fundamentals, same DCF/comps/scenario intrinsic values) must
    produce a lower or equal score, and paying less must produce a higher
    or equal score -- the rating should emerge naturally from the
    valuation/expected-return formulas as price moves, not be hard-coded.
    """
    score = compute_investment_score(**full_score_inputs)
    business_quality_points = sum(
        c.points for c in score.categories
        if c.name in ("Growth", "Profitability", "Financial Health", "Cash Flow Quality")
    )
    risk_points = next(c.points for c in score.categories if c.name == "Risk")

    table = price_sensitivity_table(
        income_df=full_score_inputs["income_df"], fcf_df=full_score_inputs["fcf_df"],
        dcf_result=full_score_inputs["dcf_result"], comps_result=full_score_inputs["comps_result"],
        scenario_results=full_score_inputs["scenario_results"], historical_pe=full_score_inputs["historical_pe"],
        dilution_rate=full_score_inputs["dilution_rate"], eps_ttm=full_score_inputs["eps_ttm"],
        ebitda_ttm=full_score_inputs["ebitda_ttm"], revenue_ttm=full_score_inputs["revenue_ttm"],
        business_quality_points=business_quality_points, risk_points=risk_points,
    )

    scores_by_price = table.sort_values("Price")["Score"].tolist()
    # Monotonically non-increasing as price rises (cheaper price -> higher score).
    assert all(earlier >= later for earlier, later in zip(scores_by_price, scores_by_price[1:]))
    # And the score must actually move -- not be flat across a +/-20% price range.
    assert scores_by_price[0] > scores_by_price[-1]


def test_price_sensitivity_table_at_current_price_matches_compute_investment_score(full_score_inputs):
    """The 'Change: +0%' row of the price-sensitivity table must reproduce
    the real score exactly (both come from the same underlying
    _price_dependent_categories function) -- otherwise the table would be
    misleading about what the actual score is."""
    score = compute_investment_score(**full_score_inputs)
    business_quality_points = sum(
        c.points for c in score.categories
        if c.name in ("Growth", "Profitability", "Financial Health", "Cash Flow Quality")
    )
    risk_points = next(c.points for c in score.categories if c.name == "Risk")

    table = price_sensitivity_table(
        income_df=full_score_inputs["income_df"], fcf_df=full_score_inputs["fcf_df"],
        dcf_result=full_score_inputs["dcf_result"], comps_result=full_score_inputs["comps_result"],
        scenario_results=full_score_inputs["scenario_results"], historical_pe=full_score_inputs["historical_pe"],
        dilution_rate=full_score_inputs["dilution_rate"], eps_ttm=full_score_inputs["eps_ttm"],
        ebitda_ttm=full_score_inputs["ebitda_ttm"], revenue_ttm=full_score_inputs["revenue_ttm"],
        business_quality_points=business_quality_points, risk_points=risk_points,
        price_multipliers=(1.0,),
        # The table must be given the SAME quality inputs the real score used.
        # The growth-adjusted valuation subscores apply a ROIC-based quality
        # adjustment, so omitting ROIC here silently produced a different
        # number in the table than in the headline score -- this assertion is
        # what caught that, and is why these are passed through.
        roe=full_score_inputs["roe"], roic=full_score_inputs["roic"],
    )
    assert table.iloc[0]["Score"] == pytest.approx(round(score.total), abs=1)


# ---------------------------------------------------------------------------
# Explicit null-vs-zero-vs-negative test cases for Expected Annualized
# Return, per the project's data-integrity requirement: a missing value
# must never be silently displayed or scored as if it were an actual zero,
# and a genuinely negative value must never be floored to zero or "N/A".
# ---------------------------------------------------------------------------

def _expected_return_case(price=100.0, fcf_ttm=50.0, shares=10.0, avg_forecast_growth=0.10,
                           target_ev_ebitda=20.0, peer_median_ev_ebitda=20.0, forecast_years=5,
                           dilution_rate=0.0, op_margin_std=0.02, eps_growth_std=0.05):
    cat, expected_return = _score_expected_return(
        price=price, fcf_ttm=fcf_ttm, shares_outstanding=shares, avg_forecast_growth=avg_forecast_growth,
        target_ev_ebitda=target_ev_ebitda, peer_median_ev_ebitda=peer_median_ev_ebitda,
        forecast_years=forecast_years, dilution_rate=dilution_rate,
        op_margin_std=op_margin_std, eps_growth_std=eps_growth_std,
    )
    sub = next(s for s in cat.subscores if "Expected annualized return" in s.label)
    return sub, expected_return


def test_case_a_positive_expected_return():
    """Test A: Expected Return = +15% -> displayed as +15%, scores well
    above neutral, contributes positively."""
    # fcf_yield = (5/10)/100 = 5%; growth = 10%; no multiple change; no
    # dilution -> expected_return = 15%.
    sub, value = _expected_return_case(avg_forecast_growth=0.10)
    assert value == pytest.approx(0.15)
    assert sub.value == pytest.approx(0.15)
    assert not sub.missing
    assert sub.points > sub.max_points * 0.5  # scores above neutral


def test_case_b_zero_expected_return_is_not_treated_as_missing():
    """Test B: Expected Return = 0% -> displayed as 0%, scored as neutral,
    and explicitly NOT flagged as missing data."""
    # fcf_yield = 5%, growth = -5% -> exactly 0%.
    sub, value = _expected_return_case(avg_forecast_growth=-0.05)
    assert value == pytest.approx(0.0, abs=1e-9)
    assert sub.value == pytest.approx(0.0, abs=1e-9)
    assert not sub.missing


def test_case_c_negative_expected_return_is_preserved_and_scores_low():
    """Test C: Expected Return = -15% -> displayed as -15% (NOT 0%, NOT
    "N/A"), and scores near the bottom of the range, contributing
    negatively to the category."""
    # fcf_yield = 5%, growth = -20% -> -15%.
    sub, value = _expected_return_case(avg_forecast_growth=-0.20)
    assert value == pytest.approx(-0.15)
    assert sub.value == pytest.approx(-0.15)  # the real negative value, not 0 and not NaN
    assert not sub.missing
    assert sub.points < sub.max_points * 0.5  # scores below neutral, reflecting the negative return


def test_case_d_null_growth_is_missing_not_zero():
    """Test D: an unavailable input (here, forecast growth) must leave the
    composite flagged as missing/neutral -- NOT silently computed as if
    growth were 0, and NOT displayed as a literal 0.0 value."""
    sub, value = _expected_return_case(avg_forecast_growth=float("nan"))
    assert np.isnan(value)
    assert np.isnan(sub.value)
    assert sub.missing is True
    # Missing data gets exactly half credit -- neither punished nor rewarded.
    assert sub.points == pytest.approx(sub.max_points * 0.5)


def test_case_f_negative_multiple_reversion_from_valuation_compression():
    """Test F: current multiple (EV/EBITDA 40x) far above the peer median
    (25x) must produce a NEGATIVE multiple-reversion contribution --
    valuation compression is a real, negative drag on expected return."""
    sub_high_multiple, value_high = _expected_return_case(
        target_ev_ebitda=40.0, peer_median_ev_ebitda=25.0, avg_forecast_growth=0.0,
    )
    sub_flat, value_flat = _expected_return_case(
        target_ev_ebitda=25.0, peer_median_ev_ebitda=25.0, avg_forecast_growth=0.0,
    )
    # Same yield and growth in both cases; only the multiple-reversion
    # term differs, so the compression case must score strictly lower.
    assert value_high < value_flat


def test_case_g_positive_multiple_reversion_from_valuation_expansion():
    """Test G: current multiple (EV/EBITDA 15x) below the peer median
    (20x) must produce a POSITIVE multiple-reversion contribution."""
    sub_low_multiple, value_low = _expected_return_case(
        target_ev_ebitda=15.0, peer_median_ev_ebitda=20.0, avg_forecast_growth=0.0,
    )
    sub_flat, value_flat = _expected_return_case(
        target_ev_ebitda=20.0, peer_median_ev_ebitda=20.0, avg_forecast_growth=0.0,
    )
    assert value_low > value_flat


def test_defaulted_multiple_reversion_and_dilution_reduce_confidence_not_silently_ignored():
    """Regression test for the exact bug reported: when the peer EV/EBITDA
    reversion target and the dilution rate are BOTH unavailable, the
    Expected Return composite must still compute (using a documented,
    neutral 0.0 assumption for each), but the Confidence subscore must be
    measurably LOWER than an otherwise-identical case where both were
    available -- the model must not silently claim full confidence in a
    result that is partly assumed rather than fully data-backed.
    """
    cat_full_data, _ = _score_expected_return(
        price=100.0, fcf_ttm=5.0, shares_outstanding=10.0, avg_forecast_growth=0.10,
        target_ev_ebitda=20.0, peer_median_ev_ebitda=20.0, forecast_years=5,
        dilution_rate=0.02, op_margin_std=0.02, eps_growth_std=0.05,
    )
    cat_missing_components, _ = _score_expected_return(
        price=100.0, fcf_ttm=5.0, shares_outstanding=10.0, avg_forecast_growth=0.10,
        target_ev_ebitda=float("nan"), peer_median_ev_ebitda=float("nan"), forecast_years=5,
        dilution_rate=float("nan"), op_margin_std=0.02, eps_growth_std=0.05,
    )
    confidence_full = next(s for s in cat_full_data.subscores if "Confidence" in s.label)
    confidence_partial = next(s for s in cat_missing_components.subscores if "Confidence" in s.label)
    assert confidence_partial.points < confidence_full.points


# ---------------------------------------------------------------------------
# Recalibration audit: regression tests for the specific defects found when
# scoring 49 real companies across blue chips, high-growth technology, value,
# unprofitable growth, banks, REITs, energy, retail and small caps.
# ---------------------------------------------------------------------------
def test_not_applicable_redistributes_instead_of_scoring_half_credit():
    """A metric that is meaningless for a business model must not drag its
    category toward 50%. Its points are re-earned at the rate the company
    achieved on the metrics that DO apply.

    Worked from the real case: a bank's Financial Health category, where
    interest coverage, Net Debt/EBITDA and Debt/EBITDA (50 of 70 points) do
    not apply. Scoring 14 of the remaining 20 (70%) must produce 70% of 70,
    not 14 plus a flat 25.
    """
    cat = CategoryScore(name="Financial Health", max_points=70, subscores=[
        SubScore("Debt / Equity", 7.0, 10.0),
        SubScore("Current ratio", 7.0, 10.0),
        SubScore("Net Debt / EBITDA", 10.0, 20.0, not_applicable=True),
        SubScore("Debt / EBITDA", 4.0, 8.0, not_applicable=True),
        SubScore("Interest coverage", 11.0, 22.0, not_applicable=True),
    ])
    _redistribute_not_applicable(cat)
    assert cat.points == pytest.approx(0.70 * 70)
    # And the inapplicable rows render as N/A rather than showing a number
    # their points did not come from.
    for sub in cat.subscores:
        if sub.not_applicable:
            assert sub.missing is True and np.isnan(sub.value)


def test_category_with_nothing_applicable_falls_back_to_neutral():
    """With no applicable evidence there is no basis to judge either way, so
    the category sits at a neutral 50% rather than inventing a verdict."""
    cat = CategoryScore(name="Cash Flow Quality", max_points=40, subscores=[
        SubScore("FCF margin", 0.0, 14.0, not_applicable=True),
        SubScore("FCF growth", 0.0, 6.0, not_applicable=True),
        SubScore("Earnings-to-cash conversion", 0.0, 10.0, not_applicable=True),
        SubScore("FCF consistency (positive in every year)", 0.0, 10.0, not_applicable=True),
    ])
    _redistribute_not_applicable(cat)
    assert cat.points == pytest.approx(20.0)


def test_company_profiles_are_detected_from_sector():
    assert detect_company_profile("Financial Services", "Banks - Diversified").is_financial
    assert detect_company_profile("Real Estate", "REIT - Industrial").is_reit
    assert detect_company_profile("Energy", "Oil & Gas Integrated").normalize_through_cycle
    assert detect_company_profile("Technology", "Software - Infrastructure").kind == "standard"
    # A warehouse club earns ~12% gross margins by design; its scale must not
    # be the software-calibrated 0-70%.
    retail = detect_company_profile("Consumer Defensive", "Discount Stores")
    assert retail.gross_margin_scale[1] < 0.70
    # Asset-heavy, spread-based models are not judged as value traps for
    # earning structurally low returns on a large capital base.
    assert detect_company_profile("Real Estate", "REIT - Retail").roic_is_quality_signal is False
    assert detect_company_profile("Technology", "Semiconductors").roic_is_quality_signal is True


def test_implausible_dcf_output_is_reported_unavailable_not_confidently_wrong():
    """Real case: a base-case DCF valued AT&T at $107 against a $25.68 price,
    and another company's DCF implied -22,800% upside from a near-zero
    terminal cash flow. Beyond the credible bound the model has broken down,
    and reporting it as unavailable is more honest than scoring it."""
    assert _sanitize_upside(0.35) == pytest.approx(0.35)
    assert _sanitize_upside(-0.65) == pytest.approx(-0.65)   # real, kept
    assert np.isnan(_sanitize_upside(MAX_CREDIBLE_DCF_UPSIDE + 0.1))
    assert np.isnan(_sanitize_upside(-228.0))
    # Discarding the tail costs nothing where the model is trustworthy: the
    # scale already awards full credit well below the cutoff.
    assert MAX_CREDIBLE_DCF_UPSIDE > 0.30


def test_high_roic_earns_a_multiple_premium_and_low_roic_a_discount():
    """The specification's two-sided rule: a high multiple is not
    automatically bad, and a low multiple is not automatically good. Holding
    price and growth identical, only the return on capital differs."""
    kwargs = dict(target_pe=30.0, target_ev_ebitda=18.0, avg_forecast_growth=0.10,
                  historical_eps_growth=0.10)
    compounder = _score_growth_adjusted_valuation(roic=0.30, **kwargs)
    average = _score_growth_adjusted_valuation(roic=0.10, **kwargs)
    value_trap = _score_growth_adjusted_valuation(roic=0.02, **kwargs)
    assert compounder.points > average.points > value_trap.points


def test_low_roic_penalty_is_not_applied_to_asset_heavy_business_models():
    """A REIT or bank earning a structurally low return on a large capital
    base is executing its business model, not decaying -- so the value-trap
    discount must not fire for them."""
    kwargs = dict(target_ev_ebitda=18.0, avg_forecast_growth=0.10,
                  historical_eps_growth=0.10, roic=0.02)
    operating = _score_growth_adjusted_valuation(target_pe=30.0, profile=CompanyProfile(), **kwargs)
    reit = _score_growth_adjusted_valuation(
        target_pe=30.0, profile=detect_company_profile("Real Estate", "REIT - Industrial"), **kwargs)
    assert reit.points > operating.points


def test_mature_company_with_slow_revenue_but_real_eps_growth_is_not_zeroed():
    """Regression test for the audit's starkest finding: Apple scored 0.3 of
    100 on Growth-Adjusted Valuation because the denominator was a fading
    revenue forecast under 2%, so any P/E produced an off-the-scale ratio.
    Blending in per-share earnings growth -- what a PEG is classically
    defined against, and what a buyback-driven mature company actually
    delivers to owners -- restores a defensible score."""
    revenue_only = _score_growth_adjusted_valuation(
        target_pe=32.0, target_ev_ebitda=22.0, avg_forecast_growth=0.018,
        historical_eps_growth=float("nan"), roic=0.80)
    with_eps_growth = _score_growth_adjusted_valuation(
        target_pe=32.0, target_ev_ebitda=22.0, avg_forecast_growth=0.018,
        historical_eps_growth=0.074, roic=0.80)
    assert with_eps_growth.points > revenue_only.points
    assert with_eps_growth.points > 0.25 * with_eps_growth.max_points


def test_trailing_eps_growth_cannot_be_inflated_by_one_recovery_year():
    """A single rebound year must not manufacture a flattering denominator:
    trailing EPS growth is capped, so +300% and +30% score identically."""
    kwargs = dict(target_pe=20.0, target_ev_ebitda=12.0, avg_forecast_growth=0.03, roic=0.15)
    capped = _score_growth_adjusted_valuation(historical_eps_growth=0.30, **kwargs)
    absurd = _score_growth_adjusted_valuation(historical_eps_growth=3.00, **kwargs)
    assert capped.points == pytest.approx(absurd.points)


def test_small_sample_valuation_percentile_is_shrunk_toward_neutral():
    """With only ~4 annual observations the percentile is near-binary: over a
    quarter of the companies tested scored EXACTLY zero because any stock
    that has re-rated sits above all of its own history. Apple and JPMorgan
    both scored 0.0/60. Shrinkage keeps the signal while refusing to call a
    four-point sample decisive."""
    four_years = pd.Series([10.0, 12.0, 14.0, 16.0])
    pts, missing, pctile = _historical_valuation_percentile_score(99.0, four_years, max_points=45)
    assert not missing
    assert pts > 0.0, "richer than every prior year must not wipe out the subscore"
    assert pctile < 100.0, "the raw 100th percentile must be shrunk toward neutral"
    # A long history is trusted much more than a short one.
    long_history = pd.Series([10.0 + i for i in range(40)])
    _, _, long_pctile = _historical_valuation_percentile_score(999.0, long_history, max_points=45)
    assert long_pctile > pctile
    # Too few points to mean anything at all.
    assert _historical_valuation_percentile_score(99.0, pd.Series([10.0, 12.0]), max_points=45)[1] is True


def test_debt_free_company_is_not_penalized_for_having_no_debt():
    """`credit_metrics` cannot divide EBIT by an interest expense that does
    not exist, and NaN previously scored a neutral 50% -- ranking a debt-free
    balance sheet BELOW a company carrying real, merely well-covered debt.
    It affected 16 of the 49 companies audited."""
    from src.scoring import _score_financial_health, EFFECTIVELY_UNLEVERED_COVERAGE
    balance = pd.DataFrame({"debt_to_equity": [0.0], "current_ratio": [3.0]})

    unlevered = _score_financial_health(
        balance, {"net_debt_to_ebitda": -1.0, "debt_to_ebitda": 0.0,
                  "interest_coverage": float("nan"), "no_meaningful_interest_expense": True})
    unknown = _score_financial_health(
        balance, {"net_debt_to_ebitda": -1.0, "debt_to_ebitda": 0.0,
                  "interest_coverage": float("nan"), "no_meaningful_interest_expense": False})

    unlevered_cov = next(s for s in unlevered.subscores if "Interest coverage" in s.label)
    unknown_cov = next(s for s in unknown.subscores if "Interest coverage" in s.label)
    assert unlevered_cov.points == pytest.approx(unlevered_cov.max_points)
    assert unlevered_cov.value == pytest.approx(EFFECTIVELY_UNLEVERED_COVERAGE)
    assert unknown_cov.points == pytest.approx(unknown_cov.max_points * 0.5)
    assert unknown_cov.missing is True


def test_leverage_thresholds_adapt_to_capital_structure():
    """A REIT funds property with mortgages; 5x Net Debt/EBITDA is its model,
    not distress. American Tower scored 5/100 on Financial Health under a
    single industrial threshold."""
    from src.scoring import _score_financial_health
    balance = pd.DataFrame({"debt_to_equity": [1.5], "current_ratio": [1.0]})
    credit = {"net_debt_to_ebitda": 5.0, "debt_to_ebitda": 5.5, "interest_coverage": 4.0}
    industrial = _score_financial_health(balance, credit, CompanyProfile())
    reit = _score_financial_health(balance, credit, detect_company_profile("Real Estate", "REIT - Specialty"))
    assert reit.points > industrial.points


def test_growth_is_measured_per_share_and_penalized_for_margin_collapse():
    """Two explicit specification requirements: dilution must recognize the
    negative effect on existing shareholders, and growth alongside
    deteriorating margins must not score as exceptional."""
    from src.scoring import _score_growth, _growth_quality_factor
    years = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"])
    steady = pd.DataFrame({
        "revenue_growth": [0.20, 0.20, 0.20, 0.20], "eps_growth": [0.20] * 4,
        "operating_margin": [0.20, 0.21, 0.22, 0.23],
    }, index=years)
    fcf = pd.DataFrame({"fcf_growth": [0.15] * 4}, index=years)

    no_dilution = _score_growth(steady, fcf, 0.20, dilution_rate=0.0)
    heavy_dilution = _score_growth(steady, fcf, 0.20, dilution_rate=0.12)
    assert heavy_dilution.points < no_dilution.points

    collapsing = steady.copy()
    collapsing["operating_margin"] = [0.25, 0.18, 0.11, 0.05]
    assert _growth_quality_factor(collapsing) < _growth_quality_factor(steady)
    assert _score_growth(collapsing, fcf, 0.20, dilution_rate=0.0).points < no_dilution.points
    # Missing margin history must never create a penalty.
    assert _growth_quality_factor(pd.DataFrame()) == pytest.approx(1.0)


def test_expected_return_falls_back_to_earnings_yield_when_fcf_unavailable():
    """The largest single subscore in the whole score used to collapse to a
    flat neutral whenever free cash flow could not be computed -- true for
    every bank by construction, and for REITs whose capex line is absent.
    JPMorgan and Prologis both scored exactly half marks on it."""
    common = dict(price=100.0, shares_outstanding=10.0, avg_forecast_growth=0.08,
                  target_ev_ebitda=12.0, peer_median_ev_ebitda=12.0, forecast_years=5,
                  dilution_rate=0.0, op_margin_std=0.02, eps_growth_std=0.05)
    without = _score_expected_return(fcf_ttm=float("nan"), eps_ttm=float("nan"), **common)[0]
    withfallback = _score_expected_return(fcf_ttm=float("nan"), eps_ttm=7.0, **common)[0]

    sub_without = next(s for s in without.subscores if "Expected annualized return" in s.label)
    sub_with = next(s for s in withfallback.subscores if "Expected annualized return" in s.label)
    assert sub_without.missing
    assert not sub_with.missing
    assert sub_with.points > sub_without.points

    # A company whose free cash flow is genuinely negative still scores on
    # that real number: with a -5% FCF yield against +8% growth the expected
    # return is a real +3%, and it must land well BELOW the +15% the earnings
    # yield would have produced. The fallback substitutes for absent data, it
    # never rescues a company from its own negative cash flow.
    negative = _score_expected_return(fcf_ttm=-50.0, eps_ttm=7.0, **common)[0]
    sub_neg = next(s for s in negative.subscores if "Expected annualized return" in s.label)
    assert sub_neg.value == pytest.approx(-0.05 + 0.08)
    assert sub_neg.points < sub_with.points


def test_multiple_reversion_cannot_swamp_the_expected_return():
    """Peer/target EV/EBITDA ratios up to 54x were observed. Unclamped, that
    implies a +115%/yr return from multiple reversion that would dominate
    every other term in the estimate."""
    from src.scoring import MAX_ANNUAL_MULTIPLE_REVERSION
    cat, expected = _score_expected_return(
        price=100.0, fcf_ttm=1.0, shares_outstanding=10.0, avg_forecast_growth=0.05,
        target_ev_ebitda=1.0, peer_median_ev_ebitda=54.0, forecast_years=5,
        dilution_rate=0.0, op_margin_std=0.02, eps_growth_std=0.05)
    assert expected <= 0.05 + 0.001 + MAX_ANNUAL_MULTIPLE_REVERSION + 1e-9


def test_bank_is_scored_on_metrics_that_apply_to_a_bank(full_score_inputs):
    """Gross margin, ROIC, current ratio, interest coverage, EV multiples and
    a FCFF DCF are all meaningless for a lender. Scoring them at a flat
    neutral dragged every bank toward ~460 and then penalized it a second
    time through data confidence."""
    generic = compute_investment_score(**full_score_inputs)
    bank = compute_investment_score(**full_score_inputs, sector="Financial Services",
                                    industry="Banks - Diversified")

    bank_names = {c.name: c for c in bank.categories}
    for cat_name in ("Profitability", "Financial Health", "Cash Flow Quality"):
        cat = bank_names[cat_name]
        assert any(s.not_applicable for s in cat.subscores), f"{cat_name} should exclude bank-inapplicable metrics"

    # Data confidence must be computed over APPLICABLE metrics only. Counting
    # the inapplicable ones as gaps is what previously drove JPMorgan to 0.5
    # of 15 points here -- penalizing it for being a bank on top of already
    # scoring it at a flat neutral on every bank-inapplicable metric.
    conf_bank = next(s for s in bank_names["Risk"].subscores if "Data confidence" in s.label)
    all_bank_subs = [s for c in bank.categories for s in c.subscores if c.name != "Risk"]
    if_counted_as_gaps = 1.0 - sum(1 for s in all_bank_subs if s.missing) / len(all_bank_subs)
    assert conf_bank.value > if_counted_as_gaps
    assert conf_bank.value > 0.85


def test_sector_profile_is_optional_and_defaults_to_standard(full_score_inputs):
    """Callers that do not supply a sector get the standard industrial
    profile, so the scoring entry point stays backward compatible."""
    without = compute_investment_score(**full_score_inputs)
    explicit = compute_investment_score(**full_score_inputs, sector="Technology",
                                        industry="Software - Infrastructure")
    assert without.total == pytest.approx(explicit.total)
