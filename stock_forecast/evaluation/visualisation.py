"""
evaluation/visualisation.py
---------------------------
Plotting utilities for the stock-forecasting pipeline.

Functions
---------
  plot_predictions(dates, y_true, predictions_dict, cfg)
      Overlays actual vs. predicted prices for all models.

  plot_training_history(history, model_name, cfg)
      Plots train/val loss curves from the LSTM training history dict.

  plot_feature_importance(importance_series, cfg, top_n)
      Horizontal bar chart of XGBoost feature importances.

  plot_residuals(y_true, y_pred, model_name, cfg)
      Residual distribution + Q-Q-style scatter plot.

All functions save to cfg.eval.output_dir when cfg.eval.save_plots is True.
"""

import logging
import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_forecast.config import MasterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predictions vs. actuals
# ---------------------------------------------------------------------------

def plot_predictions(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    cfg: MasterConfig,
    title: str = "Stock Price Forecast",
) -> None:
    """
    Overlay actual and predicted prices for one or more models.

    Parameters
    ----------
    dates            : DatetimeIndex  — x-axis labels
    y_true           : np.ndarray     — actual prices (original scale)
    predictions_dict : dict[str, np.ndarray]  — {model_name: preds}
    cfg              : MasterConfig
    title            : str
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, y_true.flatten(), label="Actual", color="black", linewidth=1.5)

    colors = plt.cm.tab10.colors
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        ax.plot(
            dates,
            preds.flatten(),
            label=name,
            color=colors[idx % len(colors)],
            linewidth=1.2,
            linestyle="--",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Price ({cfg.data.target_col})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, "predictions.png", cfg)


# ---------------------------------------------------------------------------
# LSTM training history
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, list],
    model_name: str,
    cfg: MasterConfig,
) -> None:
    """
    Plot train and validation loss over epochs.

    Parameters
    ----------
    history    : dict with 'train_loss' and 'val_loss' lists
    model_name : str
    cfg        : MasterConfig
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="darkorange")

    ax.set_title(f"{model_name} — Training History", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    filename = f"{model_name.lower().replace(' ', '_')}_training_history.png"
    _save_or_show(fig, filename, cfg)


# ---------------------------------------------------------------------------
# XGBoost feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    cfg: MasterConfig,
    top_n: int = 20,
) -> None:
    """
    Horizontal bar chart for the top-N most important features.

    Parameters
    ----------
    importance : pd.Series  — from XGBoostModel.feature_importance()
    cfg        : MasterConfig
    top_n      : int — number of features to show
    """
    top = importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n // 2)))
    ax.barh(top.index[::-1], top.values[::-1], color="steelblue")
    ax.set_title(f"XGBoost — Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance (weight)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, "xgb_feature_importance.png", cfg)


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    cfg: MasterConfig,
) -> None:
    """
    Two-panel residual plot:
      Left  — residuals over time
      Right — residual histogram with a normal-distribution overlay

    Parameters
    ----------
    y_true     : np.ndarray — actual prices
    y_pred     : np.ndarray — predicted prices
    model_name : str
    cfg        : MasterConfig
    """
    residuals = y_true.flatten() - y_pred.flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: residuals over time
    ax1.plot(residuals, color="steelblue", linewidth=0.8)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_title(f"{model_name} — Residuals over Time")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Residual (Actual − Predicted)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: residual histogram
    ax2.hist(residuals, bins=40, color="steelblue", edgecolor="white", density=True)
    mu, sigma = residuals.mean(), residuals.std()
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    normal_pdf = (
        np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    )
    ax2.plot(x_range, normal_pdf, color="darkorange", linewidth=1.5, label="Normal PDF")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_title(f"{model_name} — Residual Distribution")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    filename = f"{model_name.lower().replace(' ', '_')}_residuals.png"
    _save_or_show(fig, filename, cfg)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, filename: str, cfg: MasterConfig) -> None:
    """Save figure to output_dir if configured, then close."""
    if cfg.eval.save_plots:
        os.makedirs(cfg.eval.output_dir, exist_ok=True)
        path = os.path.join(cfg.eval.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved: %s", path)
    plt.close(fig)
