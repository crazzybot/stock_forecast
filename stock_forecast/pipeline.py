"""
pipeline.py
-----------
Top-level orchestration of the entire stock-forecasting workflow.

The StockForecastPipeline class wires together:
  1. Data ingestion    (data/ingestion.py)
  2. Feature engineering (features/engineering.py)
  3. Preprocessing     (data/preprocessing.py)
  4. Model training    (models/lstm_model.py, models/xgb_model.py)
  5. Evaluation        (evaluation/metrics.py, evaluation/visualisation.py)

Designed for extensibility:
  - Swap in a new model by adding an entry to self._models dict.
  - Override any step by subclassing and overriding the relevant method.

Usage (quick-start with synthetic data)
---------------------------------------
>>> from stock_forecast.config import MasterConfig
>>> from stock_forecast.pipeline import StockForecastPipeline
>>> import pandas as pd, numpy as np

>>> cfg = MasterConfig()
>>> df  = pd.DataFrame(...)        # your OHLCV DataFrame
>>> pipeline = StockForecastPipeline(cfg)
>>> results  = pipeline.run(df)
>>> print(results)
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from stock_forecast.config import MasterConfig
from stock_forecast.data.ingestion import load_from_dataframe
from stock_forecast.data.preprocessing import (
    fit_scaler,
    inverse_scale_target,
    make_flat,
    make_sequences,
    scale,
    split_data,
)
from stock_forecast.evaluation.metrics import compute_all_metrics, print_metrics
from stock_forecast.evaluation.visualisation import (
    plot_feature_importance,
    plot_predictions,
    plot_residuals,
    plot_training_history,
)
from stock_forecast.features.engineering import build_features
from stock_forecast.models.lstm_model import (
    LSTMTrainer,
    StockLSTM,
    make_dataloaders,
)
from stock_forecast.models.xgb_model import XGBoostModel

logger = logging.getLogger(__name__)


class StockForecastPipeline:
    """
    End-to-end stock forecasting pipeline.

    Parameters
    ----------
    cfg : MasterConfig — the single source-of-truth configuration object.
    """

    def __init__(self, cfg: MasterConfig) -> None:
        self.cfg = cfg
        self._set_seeds(cfg.random_seed)

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the complete pipeline end-to-end.

        Parameters
        ----------
        raw_df : pd.DataFrame — raw OHLCV data (DatetimeIndex or Date column)

        Returns
        -------
        dict with keys:
          'lstm_metrics'  : dict of test-set metrics for LSTM
          'xgb_metrics'   : dict of test-set metrics for XGBoost
          'lstm_preds'    : np.ndarray — LSTM predictions (original scale)
          'xgb_preds'     : np.ndarray — XGBoost predictions (original scale)
          'y_test'        : np.ndarray — actuals (original scale)
          'test_dates'    : pd.DatetimeIndex
        """
        logger.info("=" * 60)
        logger.info("  STOCK FORECAST PIPELINE  —  %s", self.cfg.data.ticker)
        logger.info("=" * 60)

        # ── Step 1: Ingest & clean ────────────────────────────────────
        logger.info("[1/5] Ingesting & cleaning raw data …")
        clean_df = load_from_dataframe(raw_df, self.cfg.data)

        # ── Step 2: Feature engineering ──────────────────────────────
        logger.info("[2/5] Engineering features …")
        feat_df = build_features(clean_df, self.cfg.data, self.cfg.features)

        # ── Step 3: Split + scale ─────────────────────────────────────
        train_df, val_df, test_df = split_data(feat_df, self.cfg.data)

        checkpoint_path = (
            Path(self.cfg.eval.model_dir) / f"{self.cfg.data.ticker}_lstm.pt"
        )
        use_saved = self.cfg.eval.load_model and checkpoint_path.exists()

        if use_saved:
            logger.info("[3/5] Loading checkpoint: %s", checkpoint_path)
            lstm_trainer, scaler, feature_cols = LSTMTrainer.load(
                checkpoint_path, self.cfg.lstm.device
            )
            lstm_history: Dict = {}
        else:
            logger.info("[3/5] Splitting & normalising …")
            scaler, feature_cols = fit_scaler(train_df, self.cfg.data)

        train_sc = scale(train_df, scaler, feature_cols)
        val_sc   = scale(val_df,   scaler, feature_cols)
        test_sc  = scale(test_df,  scaler, feature_cols)

        # ── Step 4a: LSTM ─────────────────────────────────────────────
        if use_saved:
            logger.info("[4a/5] Running LSTM inference (loaded checkpoint) …")
            X_te, _, test_dates_lstm = make_sequences(test_sc, self.cfg.data)
            lstm_preds_scaled = lstm_trainer.predict(X_te)
        else:
            logger.info("[4a/5] Training LSTM …")
            lstm_preds_scaled, lstm_history, test_dates_lstm, lstm_trainer = self._train_lstm(
                train_sc, val_sc, test_sc, feature_cols
            )
            if self.cfg.eval.save_model:
                lstm_trainer.save(checkpoint_path, scaler, feature_cols)
        lstm_preds_returns = inverse_scale_target(
            lstm_preds_scaled, scaler, feature_cols, self.cfg.data
        )
        _, y_seq_test, _ = make_sequences(test_sc, self.cfg.data)
        y_test_returns = inverse_scale_target(
            y_seq_test, scaler, feature_cols, self.cfg.data
        )
        lstm_preds, y_test_lstm = self._returns_to_close(
            lstm_preds_returns, y_test_returns, test_df, self.cfg
        )

        # ── Step 4b: XGBoost ──────────────────────────────────────────
        # logger.info("[4b/5] Training XGBoost …")
        # xgb_preds_scaled, xgb_model, test_dates_xgb = self._train_xgb(
        #     train_sc, val_sc, test_sc, feature_cols
        # )
        # xgb_preds = inverse_scale_target(
        #     xgb_preds_scaled, scaler, feature_cols, self.cfg.data
        # )
        # _, y_flat_test, _ = make_flat(test_sc, self.cfg.data)
        # y_test_xgb = inverse_scale_target(
        #     y_flat_test, scaler, feature_cols, self.cfg.data
        # )

        # ── Live prediction: next trading day ─────────────────────────
        next_close_pred = self._predict_next(
            lstm_trainer, feat_df, scaler, feature_cols, self.cfg
        )
        logger.info(
            "Next trading day predicted close: %.4f  (last known close: %.4f)",
            next_close_pred,
            feat_df[self.cfg.data.close_col].iloc[-1],
        )

        # ── Step 5: Evaluate & visualise ─────────────────────────────
        logger.info("[5/5] Evaluating & plotting …")
        lstm_metrics = compute_all_metrics(y_test_lstm, lstm_preds)
        # xgb_metrics = compute_all_metrics(y_test_xgb, xgb_preds)

        print_metrics(lstm_metrics, model_name="LSTM")
        # print_metrics(xgb_metrics, model_name="XGBoost")

        # Visualisations (saved to output_dir)
        plot_training_history(lstm_history, "LSTM", self.cfg)
        plot_predictions(
            test_dates_lstm,
            y_test_lstm,
            {"LSTM": lstm_preds},
            self.cfg,
            title=f"{self.cfg.data.ticker} — LSTM Predictions",
        )
        # plot_predictions(
        #     test_dates_xgb,
        #     y_test_xgb,
        #     {"XGBoost": xgb_preds},
        #     self.cfg,
        #     title=f"{self.cfg.data.ticker} — XGBoost Predictions",
        # )
        plot_residuals(y_test_lstm, lstm_preds, "LSTM", self.cfg)
        # plot_residuals(y_test_xgb, xgb_preds, "XGBoost", self.cfg)

        # importance = xgb_model.feature_importance()
        # plot_feature_importance(importance, self.cfg)

        logger.info("Pipeline complete.")

        return {
            "lstm_metrics": lstm_metrics,
            # "xgb_metrics": xgb_metrics,
            "lstm_preds": lstm_preds,
            # "xgb_preds": xgb_preds,
            "y_test_lstm": y_test_lstm,
            # "y_test_xgb": y_test_xgb,
            "test_dates_lstm": test_dates_lstm,
            # "test_dates_xgb": test_dates_xgb,
            "next_close_pred": next_close_pred,
        }

    # ------------------------------------------------------------------
    # Internal step implementations (override to customise)
    # ------------------------------------------------------------------

    def _train_lstm(
        self,
        train_sc: pd.DataFrame,
        val_sc: pd.DataFrame,
        test_sc: pd.DataFrame,
        feature_cols,
    ):
        """Build sequences, instantiate, train, and predict with LSTM."""
        cfg = self.cfg
        lstm_cfg = cfg.lstm
        lstm_cfg.input_size = len(feature_cols)

        X_tr, y_tr, _ = make_sequences(train_sc, cfg.data)
        X_vl, y_vl, _ = make_sequences(val_sc, cfg.data)
        X_te, _, dates = make_sequences(test_sc, cfg.data)

        train_loader, val_loader = make_dataloaders(
            X_tr, y_tr, X_vl, y_vl,
            batch_size=lstm_cfg.batch_size,
            device=lstm_cfg.device,
        )

        model = StockLSTM(lstm_cfg, forecast_horizon=cfg.data.forecast_horizon)
        trainer = LSTMTrainer(model, lstm_cfg)
        history = trainer.fit(train_loader, val_loader)

        preds = trainer.predict(X_te)
        return preds, history, dates, trainer

    def _train_xgb(
        self,
        train_sc: pd.DataFrame,
        val_sc: pd.DataFrame,
        test_sc: pd.DataFrame,
        feature_cols,
    ):
        """Build flat matrices, train, and predict with XGBoost."""
        cfg = self.cfg

        X_tr, y_tr, _ = make_flat(train_sc, cfg.data)
        X_vl, y_vl, _ = make_flat(val_sc, cfg.data)
        X_te, _, dates = make_flat(test_sc, cfg.data)

        # Build human-readable flat feature names: col_at_t-k
        n_features = len(feature_cols)
        seq_len = cfg.data.sequence_length
        flat_names = [
            f"{feature_cols[f]}_t-{seq_len - t}"
            for t in range(seq_len)
            for f in range(n_features)
        ]

        xgb_model = XGBoostModel(cfg.xgb, feature_names=flat_names)
        xgb_model.fit(X_tr, y_tr, X_vl, y_vl)

        preds = xgb_model.predict(X_te).reshape(-1, 1)
        return preds, xgb_model, dates

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_next(
        trainer: LSTMTrainer,
        feat_df: pd.DataFrame,
        scaler: Any,
        feature_cols,
        cfg: "MasterConfig",
    ) -> float:
        """Predict the next trading day's close using the tail of the full dataset."""
        seq = cfg.data.sequence_length
        tail_sc = scaler.transform(feat_df[feature_cols].iloc[-seq:])  # (seq, n_features)
        X_live = tail_sc[np.newaxis].astype(np.float32)                # (1, seq, n_features)
        pred_return_sc = trainer.predict(X_live)                        # (1, forecast_horizon)
        pred_return = inverse_scale_target(pred_return_sc, scaler, feature_cols, cfg.data)
        last_close = float(feat_df[cfg.data.close_col].iloc[-1])
        return last_close * float(np.exp(pred_return[0, 0]))

    @staticmethod
    def _returns_to_close(
        pred_returns: np.ndarray,
        actual_returns: np.ndarray,
        test_df: pd.DataFrame,
        cfg: "MasterConfig",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct absolute close prices from log-return predictions.

        For each sample i predicting date t:
          close[t+j] = close[t-1] * exp(sum of log_returns from t to t+j)
        """
        seq = cfg.data.sequence_length
        n = pred_returns.shape[0]
        prev_closes = test_df[cfg.data.close_col].to_numpy(dtype=np.float64)[seq - 1 : seq - 1 + n]
        cum_pred = np.cumsum(pred_returns, axis=1)
        cum_actual = np.cumsum(actual_returns, axis=1)
        return (
            prev_closes.reshape(-1, 1) * np.exp(cum_pred),
            prev_closes.reshape(-1, 1) * np.exp(cum_actual),
        )

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Fix all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug("Random seeds set to %d.", seed)
