"""Leakage-conscious expanding-window ridge baseline for target prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    min_train_size: int = 500,
    test_size: int = 100,
    purge: int = 1,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Generate out-of-sample predictions using expanding-window ridge regression.

    Rows must be time-sorted and features/target must share the exact index.
    Each test block is predicted by a model trained only on earlier rows.
    Purge excludes recent training rows before each test block to reduce
    overlap risk for forward-looking labels.
    """
    if not features.index.equals(target.index):
        raise ValueError("features and target must have identical indexes")
    if min_train_size < 2 or test_size < 1 or purge < 0:
        raise ValueError("min_train_size >= 2, test_size >= 1 and purge >= 0 are required")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    x = features.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(target, errors="coerce").astype(float)
    valid = x.notna().all(axis=1) & y.notna() & np.isfinite(x).all(axis=1) & np.isfinite(y)
    valid_positions = np.flatnonzero(valid.to_numpy())
    columns = ["y_true", "prediction", "fold"]
    if len(valid_positions) <= min_train_size + purge:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=features.index.name))

    predictions: list[pd.DataFrame] = []
    fold = 0
    start = min_train_size + purge
    while start < len(valid_positions):
        train_end = start - purge
        train_positions = valid_positions[:train_end]
        test_positions = valid_positions[start : start + test_size]
        if len(train_positions) < min_train_size or len(test_positions) == 0:
            break

        x_train = x.iloc[train_positions]
        x_test = x.iloc[test_positions]
        y_train = y.iloc[train_positions]
        mean = x_train.mean(axis=0)
        scale = x_train.std(axis=0, ddof=0).replace(0, 1.0)
        train_scaled = ((x_train - mean) / scale).to_numpy()
        test_scaled = ((x_test - mean) / scale).to_numpy()

        train_design = np.column_stack([np.ones(len(train_scaled)), train_scaled])
        test_design = np.column_stack([np.ones(len(test_scaled)), test_scaled])
        penalty = np.eye(train_design.shape[1]) * alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            train_design.T @ train_design + penalty,
            train_design.T @ y_train.to_numpy(),
        )
        fold_frame = pd.DataFrame(
            {
                "y_true": y.iloc[test_positions].to_numpy(),
                "prediction": test_design @ coefficients,
                "fold": fold,
            },
            index=features.index[test_positions],
        )
        predictions.append(fold_frame)
        fold += 1
        start += test_size

    if not predictions:
        return pd.DataFrame(columns=columns, index=pd.Index([], name=features.index.name))
    return pd.concat(predictions).sort_index()


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Evaluate out-of-sample point forecasts against a zero-return baseline."""
    if not {"y_true", "prediction"}.issubset(predictions.columns):
        raise ValueError("predictions must contain y_true and prediction columns")
    frame = predictions[["y_true", "prediction"]].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "zero_baseline_mae": float("nan"),
            "directional_accuracy": float("nan"),
            "correlation": float("nan"),
        }
    error = frame["prediction"] - frame["y_true"]
    correlation = (
        float(frame["prediction"].corr(frame["y_true"]))
        if frame["prediction"].nunique() > 1 and frame["y_true"].nunique() > 1
        else float("nan")
    )
    return {
        "n": int(len(frame)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "zero_baseline_mae": float(frame["y_true"].abs().mean()),
        "directional_accuracy": float(
            (np.sign(frame["prediction"]) == np.sign(frame["y_true"])).mean()
        ),
        "correlation": correlation,
    }
