"""
evaluation/metrics.py
---------------------
Unified evaluation framework for stock-forecasting models.

Metrics implemented
-------------------
  RMSE  — Root Mean Squared Error (penalises large errors heavily)
  MAE   — Mean Absolute Error    (robust, interpretable in price units)
  MAPE  — Mean Absolute Percentage Error (scale-independent %)
  DA    — Directional Accuracy   (% of correct up/down predictions)

All functions accept raw NumPy arrays in the **original price scale**
(inverse-transformed before calling these functions).

Public API
----------
  compute_all_metrics(y_true, y_pred)  -> dict
  rmse(y_true, y_pred)                -> float
  mae(y_true, y_pred)                 -> float
  mape(y_true, y_pred)                -> float
  directional_accuracy(y_true, y_pred)-> float
  print_metrics(metrics_dict)         -> None
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt( mean( (y_true - y_pred)^2 ) )

    Parameters
    ----------
    y_true, y_pred : np.ndarray — 1-D arrays of equal length

    Returns
    -------
    float
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = mean( |y_true - y_pred| )

    Parameters
    ----------
    y_true, y_pred : np.ndarray — 1-D arrays of equal length

    Returns
    -------
    float
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).

    MAPE = 100 * mean( |y_true - y_pred| / (|y_true| + epsilon) )

    The epsilon guard prevents division-by-zero for zero-valued actuals.

    Parameters
    ----------
    y_true, y_pred : np.ndarray — 1-D arrays of equal length
    epsilon        : float      — small constant to avoid division by zero

    Returns
    -------
    float  (percentage, e.g. 2.5 means 2.5 %)
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(
        100.0 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + epsilon))
    )


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (DA) — fraction of time-steps where the
    predicted direction (up / down / flat) matches the actual direction.

    Direction is defined relative to the *previous actual value*:
      actual_dir[t]    = sign(y_true[t] - y_true[t-1])
      predicted_dir[t] = sign(y_pred[t] - y_true[t-1])

    Note: requires at least 2 elements.

    Parameters
    ----------
    y_true, y_pred : np.ndarray — 1-D arrays of equal length (≥ 2)

    Returns
    -------
    float  (proportion in [0, 1])
    """
    y_true, y_pred = _validate(y_true, y_pred)
    if len(y_true) < 2:
        logger.warning("Directional accuracy requires >= 2 samples; returning NaN.")
        return float("nan")

    actual_dir = np.sign(y_true[1:] - y_true[:-1])
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    return float(np.mean(actual_dir == pred_dir))


# ---------------------------------------------------------------------------
# Composite evaluation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the full suite of evaluation metrics.

    Parameters
    ----------
    y_true, y_pred : np.ndarray — original-scale 1-D arrays

    Returns
    -------
    dict with keys: 'rmse', 'mae', 'mape', 'directional_accuracy'
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    results = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    return results


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    Pretty-print a metrics dict to stdout and the logger.

    Parameters
    ----------
    metrics    : dict returned by compute_all_metrics()
    model_name : str — label for the printed table
    """
    header = f"{'─' * 40}\n  {model_name} — Evaluation Results\n{'─' * 40}"
    lines = [header]
    for key, value in metrics.items():
        if key == "directional_accuracy":
            lines.append(f"  {key:<24}: {value * 100:.2f} %")
        elif key == "mape":
            lines.append(f"  {key:<24}: {value:.4f} %")
        else:
            lines.append(f"  {key:<24}: {value:.6f}")
    lines.append("─" * 40)
    report = "\n".join(lines)
    print(report)
    logger.info(report)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple:
    """Flatten, cast to float64, and check shape compatibility."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return y_true, y_pred
