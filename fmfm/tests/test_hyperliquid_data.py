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


def test_candles_range_chunks_within_retention_window(monkeypatch):
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
    result = client.candles_range(
        "BTC", "1m", 0, 4999 * 60_000, max_candles_per_request=3000
    )
    assert len(calls) == 2
    assert len(result) == 2


def test_candles_range_rejects_ranges_over_documented_history_limit():
    client = HyperliquidPublicData()
    with pytest.raises(ValueError, match="latest 5000 bars"):
        client.candles_range("BTC", "1m", 0, 5000 * 60_000)


def test_candles_range_rejects_unknown_interval():
    client = HyperliquidPublicData()
    with pytest.raises(ValueError):
        client.candles_range("BTC", "1mo", 0, 60_000)


def test_funding_history_paginates_and_normalizes(monkeypatch):
    client = HyperliquidPublicData()
    calls = []

    def fake_post(payload):
        calls.append(payload)
        if len(calls) == 1:
            return [
                {"coin": "BTC", "time": i * 3_600_000, "fundingRate": "0.001", "premium": "0.01"}
                for i in range(1, 501)
            ]
        return [
            {"coin": "BTC", "time": 501 * 3_600_000, "fundingRate": "0.002", "premium": "0.02"}
        ]

    monkeypatch.setattr(client, "_post_info", fake_post)
    result = client.funding_history("BTC", 0, 502 * 3_600_000)
    assert len(calls) == 2
    assert len(result) == 501
    assert result.iloc[-1]["funding_rate"] == 0.002
    assert str(result.iloc[0]["timestamp"].tz) == "UTC"


def test_current_asset_contexts_uses_public_info_endpoint(monkeypatch):
    client = HyperliquidPublicData()
    monkeypatch.setattr(client, "_post_info", lambda payload: {"type": payload["type"]})
    assert client.meta_and_asset_contexts()["type"] == "metaAndAssetCtxs"


def test_monthly_candle_request_uses_calendar_interval(monkeypatch):
    client = HyperliquidPublicData()
    monkeypatch.setattr(client, "candles", lambda coin, interval, start_ms, end_ms: pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")],
        "open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [10], "trades": [3],
    }))
    result = client.candles_range(
        "BTC", "1M",
        int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000),
        int(pd.Timestamp("2026-03-01T00:00:00Z").timestamp() * 1000),
    )
    assert len(result) == 1


def test_drop_unclosed_candles_removes_current_bar():
    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"]),
        "open": [100, 101], "high": [101, 102], "low": [99, 100],
        "close": [100, 101], "volume": [1, 1], "trades": [1, 1],
    })
    result = HyperliquidPublicData.drop_unclosed_candles(
        bars, "1m", as_of=pd.Timestamp("2026-01-01T00:01:30Z")
    )
    assert len(result) == 1
    assert result.iloc[0]["close"] == 100
