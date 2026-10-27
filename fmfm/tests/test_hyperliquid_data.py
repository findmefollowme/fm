import pandas as pd
import pytest

from fmfm.data.hyperliquid import HyperliquidPublicData


def test_candles_normalize_and_deduplicate(monkeypatch):
    client = HyperliquidPublicData()
    monkeypatch.setattr(client, "_post_info", lambda payload: [
        {"t": 60_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "n": 3},
        {"t": 60_000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "n": 3},
    ])
    result = client.candles("BTC", "1m", 0, 120_000)
    assert len(result) == 1
    assert result.iloc[0]["close"] == 1.5
    assert str(result.iloc[0]["timestamp"].tz) == "UTC"


def test_candles_range_chunks_long_ranges(monkeypatch):
    client = HyperliquidPublicData()
    calls = []

    def fake_post(payload):
        calls.append(payload)
        start = payload["req"]["startTime"]
        return [{
            "t": start, "o": "1", "h": "2", "l": "0.5",
            "c": "1.5", "v": "10", "n": 3,
        }]

    monkeypatch.setattr(client, "_post_info", fake_post)
    result = client.candles_range("BTC", "1m", 0, 6000 * 60_000, max_candles_per_request=5000)
    assert len(calls) == 2
    assert len(result) == 2


def test_candles_range_rejects_unknown_interval():
    client = HyperliquidPublicData()
    with pytest.raises(ValueError):
        client.candles_range("BTC", "1mo", 0, 60_000)
