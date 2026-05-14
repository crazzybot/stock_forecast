"""
data/ingestion.py
-----------------
Responsible for loading raw OHLCV data from CSV / parquet files
and performing initial cleaning:
  - Parse & sort dates
  - Remove fully-duplicate rows
  - Forward-fill then back-fill residual missing values
  - Enforce correct dtypes
  - Validate OHLCV sanity checks (High >= Low, Volume >= 0)

No feature engineering or normalization happens here; this module
outputs a clean, raw DataFrame ready for the preprocessing pipeline.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stock_forecast.config import DataConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(cfg: DataConfig) -> pd.DataFrame:
    """
    Load raw OHLCV data from disk.

    Supports CSV (.csv) and Parquet (.parquet / .pq) formats.
    The file path is resolved from cfg.raw_data_path with the
    ticker substituted in.

    Parameters
    ----------
    cfg : DataConfig

    Returns
    -------
    pd.DataFrame
        Clean OHLCV DataFrame indexed by date (DatetimeIndex).

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    ValueError
        If required columns are missing after loading.
    """
    path = Path(cfg.raw_data_path.format(ticker=cfg.ticker))
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info("Loaded %d rows from %s", len(df), path)
    return _clean(df, cfg)


def load_from_dataframe(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Accept an already-in-memory DataFrame (e.g. from yfinance) and
    apply the same cleaning pipeline as load_raw_data.

    Parameters
    ----------
    df  : pd.DataFrame  — raw OHLCV data
    cfg : DataConfig

    Returns
    -------
    pd.DataFrame — cleaned OHLCV DataFrame indexed by DatetimeIndex
    """
    return _clean(df.copy(), cfg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Apply all cleaning steps in order."""
    df = _parse_dates(df, cfg)
    df = _validate_columns(df, cfg)
    df = _remove_duplicates(df)
    df = _cast_dtypes(df, cfg)
    df = _handle_missing(df)
    df = _sanity_checks(df, cfg)
    return df


def _parse_dates(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Parse the date column and set it as the index, sorted ascending."""
    if cfg.date_col in df.columns:
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
        df = df.set_index(cfg.date_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        pass  # already indexed by date
    else:
        raise ValueError(
            f"Date column '{cfg.date_col}' not found and index is not DatetimeIndex."
        )
    df = df.sort_index(ascending=True)
    df.index.name = "Date"
    return df


def _validate_columns(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Ensure all required OHLCV columns are present."""
    required = [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.volume_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Keep only OHLCV to avoid carrying unknown columns forward
    return df[required]


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with duplicate date-index entries, keeping last."""
    n_before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    dropped = n_before - len(df)
    if dropped:
        logger.warning("Dropped %d duplicate rows.", dropped)
    return df


def _cast_dtypes(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Coerce all OHLCV columns to float64."""
    ohlcv = [cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.volume_col]
    df[ohlcv] = df[ohlcv].apply(pd.to_numeric, errors="coerce")
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN values:
      1. Forward-fill (carry last known value forward — avoids lookahead).
      2. Back-fill for any leading NaNs (start of series).
    """
    n_nan = df.isnull().sum().sum()
    if n_nan:
        logger.warning("Found %d NaN values — applying ffill then bfill.", n_nan)
    df = df.ffill().bfill()
    return df


def _sanity_checks(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Log warnings for common data-quality issues:
      - High < Low (broken bar)
      - Negative Close or Volume
    Does NOT drop these rows automatically; callers may handle as needed.
    """
    broken_bars = df[df[cfg.high_col] < df[cfg.low_col]]
    if not broken_bars.empty:
        logger.warning(
            "%d rows where High < Low detected:\n%s",
            len(broken_bars),
            broken_bars.head(),
        )

    neg_close = (df[cfg.close_col] <= 0).sum()
    if neg_close:
        logger.warning("%d rows with non-positive Close price.", neg_close)

    neg_vol = (df[cfg.volume_col] < 0).sum()
    if neg_vol:
        logger.warning("%d rows with negative Volume.", neg_vol)

    return df
