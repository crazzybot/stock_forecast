"""
features/engineering.py
-----------------------
Computes all technical indicators and derived features from a clean
OHLCV DataFrame.  Every calculation uses only *past* data (no future
leakage) because the `ta` library uses standard rolling windows.

Indicators produced
-------------------
Price / volume
  - Log returns (1, 5, 10-day)
  - SMA  (10, 20, 50-day)
  - EMA  (12, 26-day)
  - Volume z-score (rolling 20-day)

Momentum
  - RSI (14-day)
  - MACD line, MACD signal, MACD histogram

Volatility
  - Bollinger Band upper, lower, width, %B
  - ATR (14-day Average True Range)

All NaN rows introduced by rolling windows are dropped at the end.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from stock_forecast.config import DataConfig, FeatureConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
) -> pd.DataFrame:
    """
    Apply the full feature-engineering pipeline to a clean OHLCV DataFrame.

    Parameters
    ----------
    df       : pd.DataFrame — clean OHLCV, DatetimeIndex
    data_cfg : DataConfig
    feat_cfg : FeatureConfig

    Returns
    -------
    pd.DataFrame with all engineered features appended, NaN rows dropped.
    """
    out = df.copy()

    out = _add_return_features(out, data_cfg, feat_cfg)
    out = _add_moving_averages(out, data_cfg, feat_cfg)
    out = _add_rsi(out, data_cfg, feat_cfg)
    out = _add_macd(out, data_cfg, feat_cfg)
    out = _add_bollinger_bands(out, data_cfg, feat_cfg)
    out = _add_atr(out, data_cfg)
    out = _add_volume_features(out, data_cfg)

    # Drop rows that still have NaN after all window-based calculations
    n_before = len(out)
    out = out.dropna()
    dropped = n_before - len(out)
    if dropped:
        logger.info("Dropped %d NaN rows after feature engineering.", dropped)

    logger.info(
        "Feature matrix ready: %d rows × %d columns.", len(out), len(out.columns)
    )
    return out


# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------

def _add_return_features(
    df: pd.DataFrame, data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> pd.DataFrame:
    """Log returns over multiple horizons — shift(1) avoids lookahead."""
    close = df[data_cfg.close_col]
    for w in feat_cfg.return_windows:
        col = f"log_return_{w}d"
        df[col] = np.log(close / close.shift(w))
    return df


def _add_moving_averages(
    df: pd.DataFrame, data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> pd.DataFrame:
    """Simple and exponential moving averages, normalised by Close."""
    close = df[data_cfg.close_col]

    for w in feat_cfg.sma_windows:
        sma = close.rolling(window=w, min_periods=w).mean()
        df[f"sma_{w}"] = sma
        df[f"close_to_sma_{w}"] = close / sma - 1  # relative position

    for w in feat_cfg.ema_windows:
        ema = close.ewm(span=w, adjust=False).mean()
        df[f"ema_{w}"] = ema
        df[f"close_to_ema_{w}"] = close / ema - 1

    return df


def _add_rsi(
    df: pd.DataFrame, data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> pd.DataFrame:
    """
    Relative Strength Index (Wilder's smoothed RS method).

    RSI = 100 - 100 / (1 + RS)
    where RS = avg_gain / avg_loss over the look-back window.
    Computed manually to avoid an extra dependency on `ta` for this
    specific indicator; logic matches the standard definition.
    """
    close = df[data_cfg.close_col]
    window = feat_cfg.rsi_window

    delta = close.diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/window)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_overbought"] = (df["rsi"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(float)
    return df


def _add_macd(
    df: pd.DataFrame, data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> pd.DataFrame:
    """
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_window)
    Histogram = MACD - Signal
    """
    close = df[data_cfg.close_col]
    fast = feat_cfg.macd_fast
    slow = feat_cfg.macd_slow
    signal_w = feat_cfg.macd_signal

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal_w, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    df["macd_crossover"] = (
        (df["macd_line"] > df["macd_signal"]).astype(float)
    )
    return df


def _add_bollinger_bands(
    df: pd.DataFrame, data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> pd.DataFrame:
    """
    Bollinger Bands: SMA ± k * rolling_std
    %B = (Close - Lower) / (Upper - Lower)  — position within the band
    Width = (Upper - Lower) / SMA            — band width normalised by SMA
    """
    close = df[data_cfg.close_col]
    window = feat_cfg.bb_window
    k = feat_cfg.bb_std

    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()

    upper = sma + k * std
    lower = sma - k * std

    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = (upper - lower) / sma
    band_range = (upper - lower).replace(0, np.nan)
    df["bb_pct_b"] = (close - lower) / band_range  # 0–1 (approx)
    return df


def _add_atr(df: pd.DataFrame, data_cfg: DataConfig, window: int = 14) -> pd.DataFrame:
    """
    Average True Range (ATR) — measure of volatility.

    TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = EMA(TR, window)
    Normalised ATR = ATR / Close (dimensionless)
    """
    high = df[data_cfg.high_col]
    low = df[data_cfg.low_col]
    close = df[data_cfg.close_col]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(span=window, adjust=False).mean()
    df["atr_norm"] = df["atr"] / close
    return df


def _add_volume_features(df: pd.DataFrame, data_cfg: DataConfig) -> pd.DataFrame:
    """
    Volume-based signals:
      - Log volume (stabilises variance)
      - Volume z-score (rolling 20-day)
      - On-Balance Volume (OBV) direction
    """
    close = df[data_cfg.close_col]
    volume = df[data_cfg.volume_col]

    df["log_volume"] = np.log1p(volume)

    roll_mean = volume.rolling(20, min_periods=20).mean()
    roll_std = volume.rolling(20, min_periods=20).std().replace(0, np.nan)
    df["volume_zscore"] = (volume - roll_mean) / roll_std

    # OBV: cumulative sum of signed volume
    direction = np.sign(close.diff(1))
    df["obv"] = (direction * volume).cumsum()
    # Normalise OBV by rolling max to keep scale bounded
    obv_scale = df["obv"].abs().rolling(50, min_periods=1).max().replace(0, np.nan)
    df["obv_norm"] = df["obv"] / obv_scale

    return df


# ---------------------------------------------------------------------------
# Utility: list all feature columns (excluding raw OHLCV)
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame, data_cfg: DataConfig) -> List[str]:
    """Return engineered feature columns (raw OHLCV columns excluded)."""
    raw_cols = [
        data_cfg.open_col,
        data_cfg.high_col,
        data_cfg.low_col,
        data_cfg.close_col,
        data_cfg.volume_col,
    ]
    return [c for c in df.columns if c not in raw_cols]
