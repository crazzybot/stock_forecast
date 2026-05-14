"""
config.py
---------
Central configuration for the stock forecasting pipeline.
All hyper-parameters, paths, and constants live here so that
every other module can import a single source-of-truth.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    ticker: str = "AAPL"
    raw_data_path: str = "data/raw/{ticker}.csv"
    processed_data_path: str = "data/processed/{ticker}.parquet"

    # Column names expected in the raw CSV
    date_col: str = "Date"
    open_col: str = "Open"
    high_col: str = "High"
    low_col: str = "Low"
    close_col: str = "Close"
    volume_col: str = "Volume"

    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio is implicitly 1 - train - val = 0.15

    # Sequence length for LSTM look-back window
    sequence_length: int = 60

    # Target column the model trains on (log return keeps the target stationary)
    target_col: str = "log_return_1d"

    # Forecast horizon (steps ahead)
    forecast_horizon: int = 1


# ---------------------------------------------------------------------------
# Feature-engineering settings
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    # RSI
    rsi_window: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_window: int = 20
    bb_std: int = 2

    # Moving averages
    sma_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    ema_windows: List[int] = field(default_factory=lambda: [12, 26])

    # Return features
    return_windows: List[int] = field(default_factory=lambda: [1, 5, 10])


# ---------------------------------------------------------------------------
# LSTM model settings
# ---------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    input_size: int = 0          # set dynamically after feature engineering
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    early_stopping: bool = False
    patience: int = 50           # early-stopping patience
    clip_grad_norm: float = 1.0
    device: str = "cpu"          # "cuda" if GPU available


# ---------------------------------------------------------------------------
# XGBoost model settings
# ---------------------------------------------------------------------------
@dataclass
class XGBConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    random_state: int = 42


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    output_dir: str = "outputs/"
    model_dir: str = "outputs/models/"
    save_plots: bool = True
    save_model: bool = True
    load_model: bool = False   # skip training and use a saved checkpoint instead
    metrics: List[str] = field(
        default_factory=lambda: ["rmse", "mae", "mape", "directional_accuracy"]
    )


# ---------------------------------------------------------------------------
# Master config (compose all sub-configs)
# ---------------------------------------------------------------------------
@dataclass
class MasterConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    random_seed: int = 42
