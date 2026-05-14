"""
models/xgb_model.py
-------------------
XGBoost baseline model for stock-price forecasting.

Wraps xgboost.XGBRegressor with:
  - Walk-forward validation-set early stopping
  - Feature-importance extraction
  - Consistent predict / fit interface matching the LSTM trainer

XGBoost serves as an interpretable, fast baseline to compare against
the LSTM's learned sequential representations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from stock_forecast.config import XGBConfig

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost regression model with early stopping and feature importance.

    Parameters
    ----------
    cfg            : XGBConfig — hyper-parameters
    feature_names  : list[str] — optional column names for importance plots

    Usage
    -----
    >>> model = XGBoostModel(cfg, feature_names=cols)
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> preds = model.predict(X_test)
    >>> importance = model.feature_importance()
    """

    def __init__(
        self,
        cfg: XGBConfig,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.cfg = cfg
        self.feature_names = feature_names
        self._model: Optional[xgb.XGBRegressor] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "XGBoostModel":
        """
        Fit the XGBoost model with early stopping on validation loss.

        Parameters
        ----------
        X_train, y_train : training features and targets
        X_val,   y_val   : validation features and targets (used for
                           early stopping only — NOT for gradient updates)

        Returns
        -------
        self  (for method chaining)
        """
        cfg = self.cfg
        self._model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            eval_metric=cfg.eval_metric,
            early_stopping_rounds=cfg.early_stopping_rounds,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=0,
        )

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best = self._model.best_iteration
        score = self._model.best_score
        logger.info(
            "XGBoost trained — best iteration: %d, best val %s: %.6f",
            best, cfg.eval_metric, score,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : np.ndarray  shape (N, n_flat_features)

        Returns
        -------
        np.ndarray  shape (N,)
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._model.predict(X)

    def feature_importance(self) -> pd.Series:
        """
        Return feature importances sorted descending.

        Uses 'weight' (number of splits) importance type.
        If feature_names were provided at construction, the Series is
        labelled accordingly.

        Returns
        -------
        pd.Series  — feature importances sorted descending
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        scores = self._model.feature_importances_

        if self.feature_names and len(self.feature_names) == len(scores):
            importance = pd.Series(scores, index=self.feature_names)
        else:
            importance = pd.Series(scores)

        return importance.sort_values(ascending=False)

    def get_booster(self) -> xgb.Booster:
        """Return the underlying xgboost.Booster for advanced inspection."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._model.get_booster()
