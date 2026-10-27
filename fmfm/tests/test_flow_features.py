import numpy as np
import pandas as pd

from fmfm.features.flow import candle_flow_features, order_book_features


def test_candle_features_are_finite_when_range_nonzero():
    bars = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0],
            "close": [102.0, 103.0, 102.5],
            "volume": [10.0, 12.0, 9.0],
        }
    )
    features = candle_flow_features(bars)
    assert features.loc[1, "signed_volume_proxy"] > 0
    assert np.isfinite(features.loc[1, "turnover_proxy"])


def test_order_book_depth_imbalance():
    snapshot = {
        "levels": [
            [{"px": "100", "sz": "5"}, {"px": "99", "sz": "3"}],
            [{"px": "101", "sz": "2"}, {"px": "102", "sz": "2"}],
        ]
    }
    features = order_book_features(snapshot)
    assert features["mid_price"] == 100.5
    assert features["depth_imbalance"] > 0
    assert features["spread_bps"] > 0
