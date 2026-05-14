"""
models/lstm_model.py
--------------------
PyTorch LSTM model for multi-step stock-price forecasting.

Architecture
~~~~~~~~~~~~
  Input  →  LSTM (stacked, with dropout between layers)
         →  LayerNorm
         →  Fully-connected head
         →  Output (forecast_horizon values)

Key design choices
~~~~~~~~~~~~~~~~~~
  - Stacked LSTM with configurable depth and hidden size.
  - Dropout on LSTM inter-layer connections (regularisation).
  - LayerNorm on the final hidden state (training stability).
  - Gradient clipping in the training loop (prevents exploding gradients).
  - Early stopping with configurable patience.
  - The trainer returns a history dict for easy plotting.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stock_forecast.config import LSTMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convert NumPy arrays to PyTorch TensorDatasets and wrap in DataLoaders.

    Parameters
    ----------
    X_train, y_train : np.ndarray — training sequences & targets
    X_val,   y_val   : np.ndarray — validation sequences & targets
    batch_size       : int
    device           : str — 'cpu' or 'cuda'

    Returns
    -------
    train_loader, val_loader : DataLoader
    """
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(device)

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    val_ds = TensorDataset(_to_tensor(X_val), _to_tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class StockLSTM(nn.Module):
    """
    Stacked LSTM network for time-series regression.

    Parameters
    ----------
    cfg : LSTMConfig
        input_size   — number of features per time-step
        hidden_size  — number of LSTM units per layer
        num_layers   — number of stacked LSTM layers
        dropout      — dropout probability between layers
    forecast_horizon : int
        Number of future time-steps to predict.
    """

    def __init__(self, cfg: LSTMConfig, forecast_horizon: int = 1) -> None:
        super().__init__()
        self.cfg = cfg
        self.forecast_horizon = forecast_horizon

        # Stacked LSTM — dropout only applied between layers (not after last)
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,  # input shape: (batch, seq_len, features)
        )

        # Layer normalisation on the final hidden state
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)

        # Dropout before the FC head (additional regularisation)
        self.dropout = nn.Dropout(p=cfg.dropout)

        # Fully-connected output head
        self.fc = nn.Linear(cfg.hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor  shape (batch, forecast_horizon)
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the representation at the last time-step
        last_hidden = lstm_out[:, -1, :]           # (batch, hidden_size)
        last_hidden = self.layer_norm(last_hidden)
        last_hidden = self.dropout(last_hidden)

        out = self.fc(last_hidden)                 # (batch, forecast_horizon)
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMTrainer:
    """
    Encapsulates the PyTorch training loop, early-stopping, and
    checkpoint saving for StockLSTM.

    Usage
    -----
    >>> trainer = LSTMTrainer(model, cfg)
    >>> history = trainer.fit(train_loader, val_loader)
    >>> preds = trainer.predict(X_test)
    """

    def __init__(self, model: StockLSTM, cfg: LSTMConfig) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate
        )
        # ReduceLROnPlateau scheduler for adaptive learning rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self._best_val_loss: float = float("inf")
        self._best_weights: Optional[dict] = None
        self._patience_counter: int = 0

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.

        Parameters
        ----------
        train_loader, val_loader : DataLoader

        Returns
        -------
        history : dict with keys 'train_loss', 'val_loss'
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._eval_epoch(val_loader)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Epoch %03d/%03d — train_loss: %.6f  val_loss: %.6f",
                epoch, self.cfg.epochs, train_loss, val_loss,
            )

            if self.cfg.early_stopping and self._check_early_stopping(val_loss):
                logger.info("Early stopping at epoch %d.", epoch)
                break

        # Restore best weights
        if self.cfg.early_stopping and self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)
            logger.info("Restored best model weights (val_loss=%.6f).", self._best_val_loss)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run inference on a NumPy array.

        Parameters
        ----------
        X : np.ndarray  shape (N, seq_len, n_features)

        Returns
        -------
        np.ndarray  shape (N, forecast_horizon)
        """
        self.model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)
        return preds.cpu().numpy()

    def save(
        self,
        path: Union[str, Path],
        scaler: Any,
        feature_cols: List[str],
    ) -> None:
        """Serialize model weights, scaler, and feature list to a checkpoint file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "lstm_cfg": self.cfg,
                "forecast_horizon": self.model.forecast_horizon,
                "scaler": scaler,
                "feature_cols": feature_cols,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> Tuple["LSTMTrainer", Any, List[str]]:
        """Load a checkpoint and return (trainer, scaler, feature_cols)."""
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg: LSTMConfig = checkpoint["lstm_cfg"]
        if device is not None:
            cfg.device = device
        model = StockLSTM(cfg, forecast_horizon=checkpoint["forecast_horizon"])
        model.load_state_dict(checkpoint["model_state"])
        trainer = cls(model, cfg)
        logger.info("Checkpoint loaded ← %s", path)
        return trainer, checkpoint["scaler"], checkpoint["feature_cols"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
            )
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        return total_loss / len(loader.dataset)

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Update best weights; return True if training should stop."""
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_weights = copy.deepcopy(self.model.state_dict())
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        return self._patience_counter >= self.cfg.patience
