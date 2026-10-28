import pytest

from fmfm.data.hyperliquid_ws import build_subscriptions


def test_public_subscriptions_include_trade_book_context_and_candle():
    result = build_subscriptions("BTC", ("trades", "l2Book", "activeAssetCtx", "candle"))
    assert result == [
        {"type": "trades", "coin": "BTC"},
        {"type": "l2Book", "coin": "BTC"},
        {"type": "activeAssetCtx", "coin": "BTC"},
        {"type": "candle", "coin": "BTC", "interval": "1m"},
    ]


def test_unknown_subscription_is_rejected():
    with pytest.raises(ValueError, match="Unsupported channels"):
        build_subscriptions("BTC", ("trades", "privateOrders"))
