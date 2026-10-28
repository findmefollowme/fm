import numpy as np
import pandas as pd

from fmfm.models.walk_forward import regression_metrics, walk_forward_ridge


def test_walk_forward_ridge_only_predicts_out_of_sample_rows():
    index = pd.date_range("2026-01-01", periods=120, freq="min", tz="UTC")
    x = np.linspace(-1, 1, len(index))
    features = pd.DataFrame({"x": x, "x2": x**2}, index=index)
    target = pd.Series(0.02 * x - 0.01 * x**2, index=index)
    predictions = walk_forward_ridge(
        features, target, min_train_size=30, test_size=10, purge=1, alpha=1.0
    )
    assert len(predictions) > 0
    assert predictions.index.min() > index[30]
    assert predictions.index.is_monotonic_increasing
    assert predictions["prediction"].notna().all()


def test_regression_metrics_compare_with_zero_baseline():
    predictions = pd.DataFrame({
        "y_true": [0.1, -0.2, 0.3],
        "prediction": [0.08, -0.1, 0.2],
    })
    metrics = regression_metrics(predictions)
    assert metrics["n"] == 3
    assert metrics["mae"] < metrics["zero_baseline_mae"]
    assert 0 <= metrics["directional_accuracy"] <= 1
