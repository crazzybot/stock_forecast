"""
data/preprocessing.py
---------------------
Transforms the clean OHLCV DataFrame produced by ingestion.py into
a model-ready dataset:

  1. Chronological train / validation / test split  (NO lookahead bias —
     the scaler is fit only on the training portion).
  2. Min-Max normalisation per feature using training statistics.
  3. Sequence-window construction for LSTM (X: [T, seq_len, n_features],
     y: [T, forecast_horizon]).
  4. Flat-feature matrix construction for XGBoost  (X: [T, n_features],
     y: [T,]).

Public helpers
--------------
  split_data(df, cfg)               -> train_df, val_df, test_df
  fit_scaler(train_df, cfg)         -> (scaler, feature_cols)
  scale(df, scaler, feature_cols)   -> scaled_df
  make_sequences(scaled_df, cfg)    -> X_seq, y_seq, dates
  make_flat(scaled_df, cfg)         -> X_flat, y_flat, dates
  inverse_scale_target(arr, scaler, feature_cols, cfg) -> original-scale array
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from stock_forecast.config import DataConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Chronological split (strict — no shuffling)
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame, cfg: DataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* chronologically into train, validation, and test sets.

    The split is index-based (row order) so that the val set always
    follows the train set, and the test set always follows the val set.
    This guarantees zero lookahead bias.

    Parameters
    ----------
    df  : pd.DataFrame  — feature-engineered DataFrame (DatetimeIndex)
    cfg : DataConfig

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    n = len(df)
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. Normalisation (fit on train only)
# ---------------------------------------------------------------------------

def fit_scaler(
    train_df: pd.DataFrame, cfg: DataConfig
) -> Tuple[MinMaxScaler, List[str]]:
    """
    Fit a MinMaxScaler on *train_df* (all numeric feature columns).

    Parameters
    ----------
    train_df     : pd.DataFrame — training split
    cfg          : DataConfig

    Returns
    -------
    scaler       : fitted MinMaxScaler
    feature_cols : list of column names that were scaled
    """
    feature_cols = [c for c in train_df.columns if train_df[c].dtype != object]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feature_cols])
    logger.info("Scaler fitted on %d features.", len(feature_cols))
    return scaler, feature_cols


def scale(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to *df*.

    Parameters
    ----------
    df           : pd.DataFrame — any split (val or test)
    scaler       : fitted MinMaxScaler
    feature_cols : list of columns to scale (must match scaler input)

    Returns
    -------
    pd.DataFrame with scaled values, same index as input.
    """
    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(df[feature_cols])
    return scaled


def inverse_scale_target(
    arr: np.ndarray,
    scaler: MinMaxScaler,
    feature_cols: List[str],
    cfg: DataConfig,
) -> np.ndarray:
    """
    Reverse the MinMax scaling for the **target column only**.

    Creates a dummy matrix with zeros everywhere except the target
    column, then extracts the inverse-transformed target values.

    Parameters
    ----------
    arr          : np.ndarray  — shape (N,) or (N, forecast_horizon)
    scaler       : fitted MinMaxScaler
    feature_cols : list[str]
    cfg          : DataConfig

    Returns
    -------
    np.ndarray in the original price scale, same shape as *arr*.
    """
    target_idx = feature_cols.index(cfg.target_col)
    n_features = len(feature_cols)
    flat = arr.reshape(-1, 1)

    dummy = np.zeros((flat.shape[0], n_features))
    dummy[:, target_idx] = flat[:, 0]

    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(arr.shape)


# ---------------------------------------------------------------------------
# 3a. Sequence construction for LSTM
# ---------------------------------------------------------------------------

def make_sequences(
    scaled_df: pd.DataFrame, cfg: DataConfig
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build overlapping look-back windows for LSTM training.

    For each position t in [seq_len, len(df) - forecast_horizon]:
      X[t] = features[t - seq_len : t]          shape (seq_len, n_features)
      y[t] = target[t : t + forecast_horizon]    shape (forecast_horizon,)

    Because we use only *past* data to predict *future* data,
    there is no lookahead bias.

    Parameters
    ----------
    scaled_df : pd.DataFrame — already scaled
    cfg       : DataConfig

    Returns
    -------
    X     : np.ndarray  shape (N, seq_len, n_features)
    y     : np.ndarray  shape (N, forecast_horizon)
    dates : DatetimeIndex — date of the *prediction point* for each sample
    """
    feature_cols = list(scaled_df.columns)
    target_idx = feature_cols.index(cfg.target_col)

    data = scaled_df.values  # (T, n_features)
    T = len(data)
    seq = cfg.sequence_length
    horizon = cfg.forecast_horizon

    X_list, y_list, date_list = [], [], []

    for t in range(seq, T - horizon + 1):
        X_list.append(data[t - seq : t, :])           # past window
        y_list.append(data[t : t + horizon, target_idx])  # future target
        date_list.append(scaled_df.index[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates = pd.DatetimeIndex(date_list)

    logger.info(
        "Sequences — X: %s, y: %s  (seq_len=%d, horizon=%d)",
        X.shape, y.shape, seq, horizon,
    )
    return X, y, dates


# ---------------------------------------------------------------------------
# 3b. Flat feature matrix for XGBoost
# ---------------------------------------------------------------------------

def make_flat(
    scaled_df: pd.DataFrame, cfg: DataConfig
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build a flat (2-D) feature matrix for tree-based models.

    Uses a single look-back window flattened into one row:
      X[t] = features[t - seq_len : t].flatten()    shape (seq_len * n_features,)
      y[t] = target[t]                               scalar

    Parameters
    ----------
    scaled_df : pd.DataFrame — already scaled
    cfg       : DataConfig

    Returns
    -------
    X     : np.ndarray  shape (N, seq_len * n_features)
    y     : np.ndarray  shape (N,)
    dates : DatetimeIndex
    """
    feature_cols = list(scaled_df.columns)
    target_idx = feature_cols.index(cfg.target_col)

    data = scaled_df.values
    T = len(data)
    seq = cfg.sequence_length
    horizon = cfg.forecast_horizon

    X_list, y_list, date_list = [], [], []

    for t in range(seq, T - horizon + 1):
        X_list.append(data[t - seq : t, :].flatten())
        y_list.append(data[t, target_idx])
        date_list.append(scaled_df.index[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    dates = pd.DatetimeIndex(date_list)

    logger.info("Flat matrix — X: %s, y: %s", X.shape, y.shape)
    return X, y, dates
