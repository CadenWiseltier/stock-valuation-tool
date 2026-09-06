"""Tests for the live-price refresh layer (src/data.py).

Every test here is offline. The source functions are the only part that
touches the network, and they are replaced with deterministic fakes -- a test
that depended on a real quote would fail on a weekend, on a rate limit, or
whenever the market simply moved, which would make the suite worse than
useless. What is tested is the logic BUILT ON those sources: fallback order,
cross-checking, rejection of junk values, and the arithmetic that keeps
market cap consistent with a changed price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import (
    LIVE_PRICE_DISAGREEMENT_TOLERANCE,
    LivePrice,
    StockDataBundle,
    _coerce_price,
    apply_live_price,
    fetch_live_price,
)


def _source(price, name="Fake", as_of=None):
    """Build a fake price source returning a fixed quote."""
    def _fn(ticker):
        return LivePrice(price=price, source=name, as_of=as_of)
    return _fn


def _dead_source(ticker):
    return None


def _exploding_source(ticker):
    raise RuntimeError("provider is down")


# ---------------------------------------------------------------------------
# _coerce_price
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (258.51, 258.51),
    ("258.51", 258.51),
    ("$258.51", 258.51),        # Nasdaq returns display-formatted prices
    ("$1,258.51", 1258.51),     # thousands separator on a 4-digit price
    ("  $258.51  ", 258.51),
])
def test_coerce_price_accepts_real_values(raw, expected):
    assert _coerce_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [
    None, "", "N/A", "N/D", "-", "--", "not a number",
    0, 0.0, "$0.00",            # zero is "no data", never a real share price
    -5.0,                       # a negative price is always bad data
    float("nan"), float("inf"),
])
def test_coerce_price_rejects_junk(raw):
    """Junk must become None, never a number.

    A zero slipping through would be read downstream as a real quote and
    produce an infinite upside, which is worse than reporting nothing.
    """
    assert _coerce_price(raw) is None


# ---------------------------------------------------------------------------
# fetch_live_price: ordering, fallback, cross-checking
# ---------------------------------------------------------------------------
def test_first_source_wins_and_second_becomes_cross_check():
    lp = fetch_live_price("AMZN", sources=[_source(100.0, "A"), _source(100.2, "B")])
    assert lp.price == 100.0
    assert lp.source == "A"
    assert lp.cross_check_price == 100.2
    assert lp.cross_check_source == "B"


def test_falls_through_to_a_working_source():
    lp = fetch_live_price("AMZN", sources=[_dead_source, _source(42.0, "B")])
    assert lp.price == 42.0
    assert lp.source == "B"
    assert lp.cross_check_price is None


def test_a_raising_source_does_not_propagate():
    """A provider outage must degrade, not crash the report.

    fetch_live_price is called mid-analysis, after the user has already
    waited for the statement fetch; letting an exception escape would
    discard a report that is otherwise complete and correct.
    """
    lp = fetch_live_price("AMZN", sources=[_exploding_source, _source(7.5, "B")])
    assert lp.price == 7.5


def test_returns_none_when_every_source_fails():
    assert fetch_live_price("AMZN", sources=[_dead_source, _exploding_source]) is None


def test_blank_ticker_short_circuits():
    assert fetch_live_price("", sources=[_source(1.0)]) is None
    assert fetch_live_price("   ", sources=[_source(1.0)]) is None


def test_cross_check_disabled_stops_after_first_hit():
    lp = fetch_live_price("AMZN", sources=[_source(100.0, "A"), _source(999.0, "B")],
                          cross_check=False)
    assert lp.price == 100.0
    assert lp.cross_check_price is None
    assert lp.disagreement is None


# ---------------------------------------------------------------------------
# Disagreement detection
# ---------------------------------------------------------------------------
def test_agreeing_sources_do_not_flag():
    lp = fetch_live_price("AMZN", sources=[_source(100.0, "A"), _source(100.1, "B")])
    assert lp.disagreement == pytest.approx(0.001)
    assert not lp.sources_disagree


def test_disagreeing_sources_flag():
    lp = fetch_live_price("AMZN", sources=[_source(100.0, "A"), _source(140.0, "B")])
    assert lp.disagreement == pytest.approx(0.40)
    assert lp.sources_disagree


def test_disagreement_is_symmetric_in_magnitude():
    """A source quoting low must flag exactly as readily as one quoting high."""
    high = fetch_live_price("X", sources=[_source(100.0, "A"), _source(110.0, "B")])
    low = fetch_live_price("X", sources=[_source(100.0, "A"), _source(90.0, "B")])
    assert high.sources_disagree and low.sources_disagree
    assert high.disagreement == pytest.approx(low.disagreement)


def test_single_source_never_claims_agreement():
    """One source is 'unknown', not 'agreed' -- these must not be conflated."""
    lp = fetch_live_price("AMZN", sources=[_source(100.0, "A"), _dead_source])
    assert lp.disagreement is None
    assert not lp.sources_disagree


def test_tolerance_boundary():
    just_inside = 100.0 * (1 + LIVE_PRICE_DISAGREEMENT_TOLERANCE * 0.9)
    just_outside = 100.0 * (1 + LIVE_PRICE_DISAGREEMENT_TOLERANCE * 1.1)
    assert not fetch_live_price("X", sources=[_source(100.0), _source(just_inside)]).sources_disagree
    assert fetch_live_price("X", sources=[_source(100.0), _source(just_outside)]).sources_disagree


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
def test_unknown_timestamp_is_not_reported_as_stale():
    """'We cannot tell how old this is' is not the same claim as 'this is old'."""
    lp = LivePrice(price=1.0, source="A", as_of=None)
    assert lp.is_stale is False
    assert lp.age is None


def test_old_quote_is_stale_and_fresh_one_is_not():
    old = LivePrice(price=1.0, source="A", as_of=pd.Timestamp.utcnow() - pd.Timedelta(hours=5))
    fresh = LivePrice(price=1.0, source="A", as_of=pd.Timestamp.utcnow() - pd.Timedelta(seconds=30))
    assert old.is_stale
    assert not fresh.is_stale


# ---------------------------------------------------------------------------
# apply_live_price: keeping the bundle internally consistent
# ---------------------------------------------------------------------------
def _bundle(price=100.0, shares=1_000.0, market_cap=100_000.0, ev=120_000.0):
    return StockDataBundle(ticker="TEST", info={
        "currentPrice": price, "regularMarketPrice": price,
        "sharesOutstanding": shares, "marketCap": market_cap, "enterpriseValue": ev,
    })


def test_price_is_overwritten():
    b = apply_live_price(_bundle(), LivePrice(price=110.0, source="A"))
    assert b.info["currentPrice"] == 110.0
    assert b.info["regularMarketPrice"] == 110.0
    assert b.info["livePriceSource"] == "A"


def test_market_cap_recomputed_from_share_count():
    b = apply_live_price(_bundle(price=100.0, shares=1_000.0), LivePrice(price=110.0, source="A"))
    assert b.info["marketCap"] == pytest.approx(110_000.0)


def test_enterprise_value_moves_by_the_same_amount_holding_net_debt():
    """EV = market cap + net debt. Only the equity leg may move.

    Net debt here is 120,000 - 100,000 = 20,000 and must survive a price
    change untouched: a re-quote is not a debt repayment.
    """
    b = apply_live_price(_bundle(market_cap=100_000.0, ev=120_000.0),
                         LivePrice(price=110.0, source="A"))
    assert b.info["enterpriseValue"] == pytest.approx(130_000.0)
    assert b.info["enterpriseValue"] - b.info["marketCap"] == pytest.approx(20_000.0)


def test_market_cap_scales_proportionally_when_share_count_is_missing():
    b = StockDataBundle(ticker="T", info={
        "currentPrice": 100.0, "marketCap": 100_000.0, "enterpriseValue": 120_000.0,
    })
    apply_live_price(b, LivePrice(price=150.0, source="A"))
    assert b.info["marketCap"] == pytest.approx(150_000.0)


def test_none_live_price_leaves_the_bundle_untouched():
    """No quote must mean 'unchanged', never 'zeroed' or 'guessed'."""
    b = _bundle()
    before = dict(b.info)
    apply_live_price(b, None)
    assert b.info == before


def test_missing_market_cap_does_not_invent_one():
    b = StockDataBundle(ticker="T", info={"currentPrice": 100.0})
    apply_live_price(b, LivePrice(price=110.0, source="A"))
    assert b.info["currentPrice"] == 110.0
    assert "marketCap" not in b.info


def test_applying_a_price_is_idempotent():
    """Re-applying the same quote must not compound the market-cap change."""
    b = _bundle()
    lp = LivePrice(price=110.0, source="A")
    apply_live_price(b, lp)
    first = b.info["marketCap"]
    apply_live_price(b, lp)
    assert b.info["marketCap"] == pytest.approx(first)
