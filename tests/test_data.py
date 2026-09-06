"""Unit tests for src/data.py: the business-summary sector override used for
peer selection, and the distinction between a provider failure and a bad
ticker."""

import pandas as pd
import pytest

from src.data import (
    DataProviderError, TickerNotFoundError, _describe_provider_failure, infer_peer_sector,
)


def test_sector_override_fires_for_data_center_company_misclassified_as_financial():
    """Regression test for a real bug report: IREN Limited (an AI cloud /
    data-center operator) is classified by the data provider as
    "Financial Services", which produces a peer group of megabanks --
    an economically meaningless comparison. The override should detect
    the mismatch from the company's own reported business description and
    substitute "Technology" for peer-selection purposes.
    """
    info = {
        "sector": "Financial Services",
        "longBusinessSummary": (
            "IREN Limited operates as a vertically integrated AI cloud services "
            "platform. It develops, owns, and operates data centers, including "
            "the associated land, grid connections, and compute layers."
        ),
    }
    assert infer_peer_sector(info) == "Technology"


def test_sector_override_does_not_fire_when_reported_sector_already_matches():
    """A company genuinely in the inferred sector should not trigger the
    override machinery (there is nothing to correct)."""
    info = {
        "sector": "Technology",
        "longBusinessSummary": "Operates data centers and cloud computing infrastructure.",
    }
    assert infer_peer_sector(info) == "Technology"


def test_sector_override_does_not_fire_without_a_matching_keyword():
    """A normal, correctly-classified company must pass through unchanged --
    this is a narrow override, not a general reclassification system."""
    info = {
        "sector": "Consumer Defensive",
        "longBusinessSummary": "Manufactures and sells packaged food and beverage products.",
    }
    assert infer_peer_sector(info) == "Consumer Defensive"


def test_sector_override_handles_missing_business_summary():
    info = {"sector": "Financial Services"}
    assert infer_peer_sector(info) == "Financial Services"


# ---------------------------------------------------------------------------
# Provider-failure vs. bad-ticker: two failures that need opposite advice.
# ---------------------------------------------------------------------------
def test_rate_limit_failure_is_described_as_rate_limiting():
    """A throttled request must not be reported as a bad ticker. On a public
    deployment every visitor shares one server IP, so rate limiting is the
    most likely failure mode -- and telling someone to "check the ticker
    symbol" when they typed AAPL correctly makes the site look broken."""
    msg = _describe_provider_failure([Exception("429 Client Error: Too Many Requests for url ...")])
    assert "rate-limiting" in msg.lower()
    assert "ticker" in msg.lower(), "must reassure the user their ticker was not the problem"


def test_connectivity_failure_is_described_as_a_network_problem():
    msg = _describe_provider_failure([Exception("curl: (6) Could not resolve host: query2.finance.yahoo.com")])
    assert "reach" in msg.lower() or "network" in msg.lower()
    assert "rate-limiting" not in msg.lower()


def test_unrecognized_provider_failure_still_gets_a_usable_message():
    msg = _describe_provider_failure([Exception("something entirely unexpected")])
    assert msg and "try again" in msg.lower()


def test_provider_error_and_ticker_error_are_distinct_types():
    """They are caught separately in app.py and shown differently, so one must
    never be mistaken for the other."""
    assert not issubclass(DataProviderError, TickerNotFoundError)
    assert not issubclass(TickerNotFoundError, DataProviderError)


def test_provider_failure_raises_provider_error_not_ticker_error(monkeypatch):
    """End-to-end: when the provider throws on every call, `fetch_stock_data`
    must report a provider problem rather than claiming the ticker is
    invalid."""
    import src.data as data_module

    class _ThrowingTicker:
        def __init__(self, symbol):
            pass

        def get_info(self):
            raise Exception("429 Too Many Requests")

        def history(self, **kwargs):
            raise Exception("429 Too Many Requests")

    monkeypatch.setattr(data_module.yf, "Ticker", _ThrowingTicker)
    with pytest.raises(DataProviderError) as excinfo:
        data_module.fetch_stock_data("AAPL")
    assert "rate-limiting" in str(excinfo.value).lower()


def test_genuinely_unknown_ticker_still_raises_ticker_not_found(monkeypatch):
    """The provider answered cleanly and had nothing -- that IS a bad ticker,
    and must keep saying so."""
    import src.data as data_module

    class _EmptyTicker:
        def __init__(self, symbol):
            pass

        def get_info(self):
            return {}

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(data_module.yf, "Ticker", _EmptyTicker)
    with pytest.raises(TickerNotFoundError):
        data_module.fetch_stock_data("ZZZZNOTAREALTICKER")
