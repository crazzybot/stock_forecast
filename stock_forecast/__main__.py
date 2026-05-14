import argparse
import logging
import sys

from stock_forecast.config import MasterConfig
from stock_forecast.data.ingestion import load_raw_data
from stock_forecast.pipeline import StockForecastPipeline

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock forecast pipeline")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    cfg = MasterConfig()
    cfg.data.ticker = args.ticker

    raw_df = load_raw_data(cfg.data)
    pipeline = StockForecastPipeline(cfg)
    results = pipeline.run(raw_df)
    print(results)