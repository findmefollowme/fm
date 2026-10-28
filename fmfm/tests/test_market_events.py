import pandas as pd

from fmfm.data.events import asset_contexts_from_events, book_features_from_events, trades_from_events
from fmfm.features.flow import trade_flow_features


def _event(channel, payload, coin="BTC"):
    return {
        "coin": coin,
        "channel": channel,
        "ingest_time": "2026-01-01T00:01:00Z",
        "payload": {"channel": channel, "data": payload},
    }


def test_normalize_trade_events_and_aggregate_at_minute_close():
    events = [_event("trades", [
        {"coin": "BTC", "time": 10_000, "side": "B", "sz": "2", "px": "100", "tid": 1},
        {"coin": "BTC", "time": 30_000, "side": "A", "sz": "1", "px": "101", "tid": 2},
    ])]
    trades = trades_from_events(events, coin="BTC")
    assert len(trades) == 2
    features = trade_flow_features(trades)
    assert features.index[0] == pd.Timestamp("1970-01-01T00:01:00Z")
    assert features.iloc[0]["net_aggressive_notional"] == 99
    assert features.iloc[0]["total_notional"] == 301


def test_normalize_l2_book_events():
    snapshot = {
        "coin": "BTC",
        "time": 1_767_225_600_000,
        "levels": [
            [{"px": "100", "sz": "5"}],
            [{"px": "101", "sz": "2"}],
        ],
    }
    result = book_features_from_events([_event("l2Book", snapshot)])
    assert len(result) == 1
    assert result.iloc[0]["spread_bps"] > 0
    assert result.iloc[0]["depth_imbalance"] > 0


def test_normalize_asset_context_snapshots():
    context = {
        "coin": "BTC",
        "ctx": {"funding": "0.001", "openInterest": "123", "markPx": "100"},
    }
    result = asset_contexts_from_events([_event("activeAssetCtx", context)])
    assert len(result) == 1
    assert result.iloc[0]["funding"] == 0.001
    assert result.iloc[0]["openInterest"] == 123
