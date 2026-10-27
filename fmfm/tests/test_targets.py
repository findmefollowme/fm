import numpy as np
import pandas as pd

from fmfm.targets import forward_log_return, forward_realized_volatility


def test_forward_log_return_is_future_only():
    close = pd.Series([100.0, 110.0, 121.0])
    got = forward_log_return(close, 1)
    assert np.isclose(got.iloc[0], np.log(1.1))
    assert np.isnan(got.iloc[-1])


def test_forward_volatility_uses_next_returns():
    close = pd.Series([100.0, 110.0, 99.0, 108.9])
    got = forward_realized_volatility(close, 2)
    expected = np.sqrt(np.log(0.9) ** 2 + np.log(1.1) ** 2)
    assert np.isclose(got.iloc[1], expected)
    assert np.isnan(got.iloc[-1])
