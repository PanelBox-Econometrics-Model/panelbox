"""Forecast evaluation metrics for comparing predictions."""

import numpy as np
import pandas as pd


def rmse(actual, predicted) -> float:
    """
    Root Mean Squared Error.

    Parameters
    ----------
    actual : array-like
        Observed values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted) -> float:
    """
    Mean Absolute Error.

    Parameters
    ----------
    actual : array-like
        Observed values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual, predicted) -> float:
    """
    Mean Absolute Percentage Error.

    Parameters
    ----------
    actual : array-like
        Observed values (non-zero).
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE value (0-100 scale).
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def theil_u(actual, predicted) -> float:
    """
    Theil's U statistic.

    U = 1 means forecast is no better than random walk.
    U < 1 means forecast beats random walk.

    Parameters
    ----------
    actual : array-like
        Observed values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        Theil's U value.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) < 2:
        return np.nan
    # Naive forecast = previous actual
    naive = actual[:-1]
    actual_diff = actual[1:]
    pred_diff = predicted[1:]

    rmse_model = np.sqrt(np.mean((actual_diff - pred_diff) ** 2))
    rmse_naive = np.sqrt(np.mean((actual_diff - naive) ** 2))

    if rmse_naive == 0:
        return np.nan
    return float(rmse_model / rmse_naive)


def direction_accuracy(actual, predicted) -> float:
    """
    Percentage of correctly predicted directions (up/down).

    Parameters
    ----------
    actual : array-like
        Observed values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        Direction accuracy (0-100 scale).
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if len(actual) < 2:
        return np.nan
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    return float(np.mean(actual_dir == pred_dir) * 100)


def forecast_evaluation_table(
    actual: np.ndarray,
    forecasts: dict,
) -> pd.DataFrame:
    """
    Create comparison table of multiple models.

    Parameters
    ----------
    actual : array-like
        Observed values.
    forecasts : dict
        Mapping {model_name: predictions_array}.

    Returns
    -------
    pd.DataFrame
        Table with RMSE, MAE, MAPE, Theil-U, Direction columns per model.
    """
    actual = np.asarray(actual)
    rows = []
    for name, preds in forecasts.items():
        preds = np.asarray(preds)
        rows.append(
            {
                "Model": name,
                "RMSE": rmse(actual, preds),
                "MAE": mae(actual, preds),
                "MAPE": mape(actual, preds),
                "Theil-U": theil_u(actual, preds),
                "Direction (%)": direction_accuracy(actual, preds),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def prediction_interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Compute coverage rate of prediction intervals.

    Parameters
    ----------
    actual : array-like
        Observed values.
    lower : array-like
        Lower bound of prediction interval.
    upper : array-like
        Upper bound of prediction interval.

    Returns
    -------
    float
        Coverage rate (0-1 scale).
    """
    actual = np.asarray(actual)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    covered = (actual >= lower) & (actual <= upper)
    return float(np.mean(covered))
