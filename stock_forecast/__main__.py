import argparse
import logging
import sys

from stock_forecast.config import MasterConfig
from stock_forecast.data.ingestion import load_raw_data
from stock_forecast.pipeline import StockForecastPipeline

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock forecast pipeline")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load pre-trained model instead of training a new one",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping during training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    cfg = MasterConfig()
    cfg.data.ticker = args.ticker
    cfg.eval.load_model = args.load_model
    cfg.lstm.early_stopping = args.early_stopping
    cfg.lstm.epochs = args.epochs
    raw_df = load_raw_data(cfg.data)
    pipeline = StockForecastPipeline(cfg)
    results = pipeline.run(raw_df)
    # print(results)
