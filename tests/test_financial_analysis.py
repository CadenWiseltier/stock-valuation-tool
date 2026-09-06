"""Unit tests for src/financial_analysis.py, using the synthetic 3-year
company fixture in tests/conftest.py, whose values are hand-computable."""

import numpy as np
import pandas as pd
import pytest

from src.data import StockDataBundle
from src.financial_analysis import (
    balance_sheet_summary, credit_metrics, earnings_to_cash_conversion,
    fcf_summary, income_statement_summary, is_financial_services_company,
    roe_series, roic_series,
)


def test_income_statement_margins_and_growth(synthetic_bundle):
    df = income_statement_summary(synthetic_bundle)

    assert df["gross_margin"].iloc[0] == pytest.approx(0.60)
    assert df["operating_margin"].iloc[0] == pytest.approx(0.20)
    assert df["net_margin"].iloc[0] == pytest.approx(0.15)

    # Revenue grows 1000 -> 1100 -> 1210, i.e. exactly 10% each year.
    assert np.isnan(df["revenue_growth"].iloc[0])
    assert df["revenue_growth"].iloc[1] == pytest.approx(0.10)
    assert df["revenue_growth"].iloc[2] == pytest.approx(0.10)

    # EPS grows 1.50 -> 1.65 -> 1.815, also exactly 10% each year.
    assert df["eps_growth"].iloc[1] == pytest.approx(0.10)
    assert df["eps_growth"].iloc[2] == pytest.approx(0.10, rel=1e-3)


def test_free_cash_flow(synthetic_bundle):
    df = fcf_summary(synthetic_bundle)

    # FCF = OCF - CapEx: 220-50=170, 240-55=185, 264-60=204
    assert df["free_cash_flow"].iloc[0] == pytest.approx(170.0)
    assert df["free_cash_flow"].iloc[1] == pytest.approx(185.0)
    assert df["free_cash_flow"].iloc[2] == pytest.approx(204.0)

    assert df["fcf_margin"].iloc[0] == pytest.approx(0.17)
    assert df["fcf_growth"].iloc[1] == pytest.approx(185 / 170 - 1)


def test_earnings_to_cash_conversion(synthetic_bundle):
    conv = earnings_to_cash_conversion(synthetic_bundle)
    # Year 0: OCF 220 / Net Income 150
    assert conv.iloc[0] == pytest.approx(220 / 150)


def test_balance_sheet_ratios(synthetic_bundle):
    df = balance_sheet_summary(synthetic_bundle)

    assert df["net_debt"].iloc[0] == pytest.approx(100.0)   # 300 - 200
    assert df["net_debt"].iloc[2] == pytest.approx(0.0)     # 300 - 300
    assert df["current_ratio"].iloc[0] == pytest.approx(500 / 250)
    assert df["debt_to_equity"].iloc[0] == pytest.approx(300 / 800)


def test_credit_metrics(synthetic_bundle):
    metrics = credit_metrics(synthetic_bundle)
    # Latest year (2023): EBITDA 314.6, EBIT 242.0, Interest Expense 10.0,
    # Total Debt 300, Cash 300.
    assert metrics["debt_to_ebitda"] == pytest.approx(300 / 314.6, rel=1e-4)
    assert metrics["net_debt_to_ebitda"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["interest_coverage"] == pytest.approx(242 / 10, rel=1e-4)


def test_roe(synthetic_bundle):
    roe = roe_series(synthetic_bundle)
    # Year 0 has no prior-period equity, so average equity falls back to
    # period-end equity: ROE = 150 / 800.
    assert roe.iloc[0] == pytest.approx(150 / 800)
    # Year 1: average equity = (800 + 900) / 2 = 850; ROE = 165 / 850.
    assert roe.iloc[1] == pytest.approx(165 / 850)


def test_roic_is_computed_and_reasonable(synthetic_bundle):
    roic = roic_series(synthetic_bundle)
    assert not roic.dropna().empty
    # ROIC should be a sensible fraction (not negative, not absurdly large)
    # for a profitable, moderately levered company like the fixture.
    for v in roic.dropna():
        assert 0.0 < v < 1.0


def test_eps_growth_from_a_loss_year_is_masked_not_a_huge_percentage():
    """Regression test for a real bug report: a company posting a small
    loss one year and a real profit the next (e.g. AMZN's -$0.27 -> $2.90
    EPS transition) must NOT report that as a literal "-1174%" growth
    figure -- that number is not economically meaningful (the sign of the
    denominator flips what "growth" even means) and, left in, dominates
    and distorts every downstream volatility statistic computed from the
    series. It should be masked to NaN ("Not Meaningful"), not merely a
    very negative number.
    """
    years = [pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    income_stmt = pd.DataFrame({
        years[0]: {"Total Revenue": 500.0, "Diluted EPS": -0.27},
        years[1]: {"Total Revenue": 550.0, "Diluted EPS": 2.90},
        years[2]: {"Total Revenue": 600.0, "Diluted EPS": 3.20},
    })
    bundle = StockDataBundle(ticker="TEST", info={}, income_stmt=income_stmt,
                              balance_sheet=pd.DataFrame(), cash_flow=pd.DataFrame(),
                              price_history=pd.DataFrame())

    df = income_statement_summary(bundle)

    # The loss -> profit transition must be masked, not a huge negative number.
    assert np.isnan(df["eps_growth"].iloc[1])
    # The following (profit -> profit) transition is unaffected and still computed normally.
    assert df["eps_growth"].iloc[2] == pytest.approx(3.20 / 2.90 - 1)


def test_fcf_growth_from_a_negative_fcf_year_is_masked():
    """Same masking rule applied to FCF growth, since FCF can also
    genuinely cross zero (e.g. a heavy capex year)."""
    years = [pd.Timestamp("2022-12-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    cash_flow = pd.DataFrame({
        years[0]: {"Operating Cash Flow": 50.0, "Capital Expenditure": -80.0},   # FCF = -30
        years[1]: {"Operating Cash Flow": 90.0, "Capital Expenditure": -40.0},   # FCF = +50
        years[2]: {"Operating Cash Flow": 100.0, "Capital Expenditure": -45.0},  # FCF = +55
    })
    bundle = StockDataBundle(ticker="TEST", info={}, income_stmt=pd.DataFrame(),
                              balance_sheet=pd.DataFrame(), cash_flow=cash_flow,
                              price_history=pd.DataFrame())

    df = fcf_summary(bundle)

    assert np.isnan(df["fcf_growth"].iloc[1])  # -30 -> 50: masked (negative prior base)
    assert df["fcf_growth"].iloc[2] == pytest.approx(55 / 50 - 1)  # 50 -> 55: computed normally


def _bank_bundle_with_loan_distorted_cash_flow() -> StockDataBundle:
    """A synthetic bank/lender: reported Operating Cash Flow runs to
    negative billions against modest revenue -- modeled on SOFI, whose
    cash flow statement includes loan origination/sale and deposit
    activity within "operating" cash flow (a real, well-known GAAP
    presentation quirk for banks and lenders, not a data error)."""
    years = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    income_stmt = pd.DataFrame({
        years[0]: {"Total Revenue": 2_000_000_000.0, "Net Income": -300_000_000.0},
        years[1]: {"Total Revenue": 2_600_000_000.0, "Net Income": 480_000_000.0},
    })
    cash_flow = pd.DataFrame({
        years[0]: {"Operating Cash Flow": -7_200_000_000.0, "Capital Expenditure": -120_000_000.0},
        years[1]: {"Operating Cash Flow": -1_100_000_000.0, "Capital Expenditure": -160_000_000.0},
    })
    return StockDataBundle(
        ticker="TEST", info={"sector": "Financial Services", "industry": "Credit Services"},
        income_stmt=income_stmt, balance_sheet=pd.DataFrame(), cash_flow=cash_flow,
        price_history=pd.DataFrame(),
    )


def test_fcf_is_masked_not_confidently_wrong_for_a_bank_or_lender():
    """Regression test for a real bug report: a bank/lender's reported
    Operating Cash Flow is dominated by loan origination/deposit activity,
    not discretionary reinvestment the way it is for an industrial company
    -- "OCF - CapEx" is not a meaningful measure of free cash flow for
    this class of company (found in testing: SOFI showed FCF of negative
    BILLIONS against ~$3B of revenue, which then crushed the Expected
    Return score to a confident 0 rather than an honest N/A). FCF/margin/
    growth should be masked to NaN; Operating Cash Flow and CapEx
    themselves are still real, reported figures and should NOT be masked.
    """
    bundle = _bank_bundle_with_loan_distorted_cash_flow()
    assert is_financial_services_company(bundle) is True

    df = fcf_summary(bundle)
    assert df["operating_cash_flow"].notna().all()
    assert df["capex"].notna().all()
    assert df["free_cash_flow"].isna().all()
    assert df["fcf_margin"].isna().all()
    assert df["fcf_growth"].isna().all()

    conversion = earnings_to_cash_conversion(bundle)
    assert conversion.empty or conversion.isna().all()


def test_fcf_is_computed_normally_for_a_non_financial_company(synthetic_bundle):
    """Sanity check: the masking above must not accidentally suppress FCF
    for an ordinary (non-financial-services) company."""
    assert is_financial_services_company(synthetic_bundle) is False
    df = fcf_summary(synthetic_bundle)
    assert df["free_cash_flow"].notna().any()


def _negative_book_equity_bundle() -> StockDataBundle:
    """A mature, genuinely profitable company with NEGATIVE book equity
    from a history of aggressive share buybacks -- modeled on McDonald's
    and Starbucks, both confirmed in manual testing. Net income and EBIT
    are solidly positive; only shareholders' equity is negative."""
    years = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    income_stmt = pd.DataFrame({
        years[0]: {"Net Income": 8_000_000_000.0, "EBIT": 11_000_000_000.0,
                   "Pretax Income": 10_000_000_000.0, "Tax Provision": 2_000_000_000.0},
        years[1]: {"Net Income": 8_300_000_000.0, "EBIT": 11_400_000_000.0,
                   "Pretax Income": 10_400_000_000.0, "Tax Provision": 2_100_000_000.0},
    })
    balance_sheet = pd.DataFrame({
        years[0]: {"Total Debt": 40_000_000_000.0, "Stockholders Equity": -4_700_000_000.0,
                   "Cash And Cash Equivalents": 2_000_000_000.0},
        years[1]: {"Total Debt": 41_000_000_000.0, "Stockholders Equity": -3_800_000_000.0,
                   "Cash And Cash Equivalents": 2_100_000_000.0},
    })
    return StockDataBundle(ticker="TEST", info={}, income_stmt=income_stmt,
                            balance_sheet=balance_sheet, cash_flow=pd.DataFrame(),
                            price_history=pd.DataFrame())


def test_roe_and_debt_to_equity_are_masked_for_negative_book_equity():
    """Regression test for a real bug report: a genuinely profitable
    company with negative book equity (McDonald's, Starbucks, confirmed in
    manual testing) must NOT have its ROE reported as a sign-flipped,
    catastrophic-looking loss (McDonald's computed as low as -307% ROE
    despite ~20% ROIC in the same year). Net Income / Negative Equity is
    not a meaningful ratio and should be masked to N/A, not presented as a
    confidently wrong (and backwards) number. Debt/Equity has the same
    issue and the same fix.
    """
    bundle = _negative_book_equity_bundle()

    roe = roe_series(bundle)
    assert roe.isna().all()

    bs = balance_sheet_summary(bundle)
    assert bs["debt_to_equity"].isna().all()

    # ROIC uses invested capital (debt + equity - cash), which stays
    # positive here even though equity alone is negative -- it should
    # still compute normally and show these companies' real, strong returns.
    roic = roic_series(bundle)
    assert roic.notna().any()
    assert (roic.dropna() > 0).all()


def _negative_ebitda_bundle() -> StockDataBundle:
    """An early-stage, capital-intensive company with genuine, ongoing
    operating losses -- modeled on Lucid Motors and Rivian, both confirmed
    in manual testing (negative EBITDA/EBIT, positive book equity)."""
    years = [pd.Timestamp("2023-12-31"), pd.Timestamp("2024-12-31")]
    income_stmt = pd.DataFrame({
        years[0]: {"EBITDA": -2_500_000_000.0, "EBIT": -2_800_000_000.0,
                   "Interest Expense": 100_000_000.0},
        years[1]: {"EBITDA": -2_400_000_000.0, "EBIT": -2_700_000_000.0,
                   "Interest Expense": 110_000_000.0},
    })
    balance_sheet = pd.DataFrame({
        years[0]: {"Total Debt": 2_200_000_000.0, "Cash And Cash Equivalents": 1_000_000_000.0},
        years[1]: {"Total Debt": 2_300_000_000.0, "Cash And Cash Equivalents": 900_000_000.0},
    })
    return StockDataBundle(ticker="TEST", info={}, income_stmt=income_stmt,
                            balance_sheet=balance_sheet, cash_flow=pd.DataFrame(),
                            price_history=pd.DataFrame())


def test_leverage_and_coverage_ratios_are_masked_for_negative_ebitda():
    """Regression test for a real bug report: a company with genuine,
    ongoing operating losses (Lucid Motors, Rivian, confirmed in manual
    testing) must NOT have Debt/EBITDA computed as a sign-flipped negative
    number (which this model's scoring scale would read as "very low
    leverage," the opposite of reality), and must NOT have Interest
    Coverage computed from abs(EBIT) (which produced a misleadingly
    reassuring 10-27x "coverage" ratio for companies burning billions in
    operating losses with no profit to cover interest from at all).
    """
    bundle = _negative_ebitda_bundle()
    credit = credit_metrics(bundle)

    assert np.isnan(credit["debt_to_ebitda"])
    assert np.isnan(credit["net_debt_to_ebitda"])
    assert np.isnan(credit["interest_coverage"])
    # Net debt itself (not earnings-relative) is unaffected and still
    # computed, using the latest available year (2024: debt 2.3B, cash 0.9B).
    assert credit["net_debt"] == pytest.approx(2_300_000_000.0 - 900_000_000.0)
