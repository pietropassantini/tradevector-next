"""Build OI features from raw data and save as parquet."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.data.alignment import align_and_validate
from tradevector.data.loaders import load_oi_data, load_klines
from tradevector.features.oi_features import build_oi_features, save_features


def main():
    parser = argparse.ArgumentParser(description="Build OI features from raw data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--oi-period", default="5m", help="OI data period")
    parser.add_argument("--hypothesis-id", default="oi_compression_breakout", help="Hypothesis ID")
    parser.add_argument("--save", action="store_true", default=True, help="Save features to parquet")
    parser.add_argument("--max-oi-staleness", default="2h",
                        help="Oltre questa età l'OI ffillato diventa NaN (default: 2h)")

    args = parser.parse_args()

    logger.info(f"Loading klines for {args.symbol} {args.timeframe}...")
    try:
        candles = load_klines(args.symbol, args.timeframe)
        logger.info(f"Loaded {len(candles)} candles")
    except FileNotFoundError:
        logger.error(
            f"Klines not found. Run downloader first:\n"
            f"  python scripts/download_open_interest_binance.py "
            f"--symbol {args.symbol} --klines-interval {args.timeframe} --download-klines"
        )
        sys.exit(1)

    logger.info(f"Loading OI data for {args.symbol}...")
    try:
        oi_data = load_oi_data(args.symbol, args.oi_period)
        logger.info(f"Loaded {len(oi_data)} OI records")
    except FileNotFoundError:
        logger.error(
            f"OI data not found. Run downloader first:\n"
            f"  python scripts/download_open_interest_binance.py --symbol {args.symbol}"
        )
        sys.exit(1)

    logger.info("Aligning OI data to candle timestamps...")
    oi_aligned = align_and_validate(
        external_df=oi_data,
        candles_df=candles,
        source_name=f"{args.symbol}_oi",
        method="last_known",
        # I buchi nella serie OI restano NaN invece di propagarsi come costante:
        # una costante ffillata annulla la dispersione delle feature derivate.
        max_lookback=pd.Timedelta(args.max_oi_staleness),
    )

    logger.info("Building features...")
    features = build_oi_features(
        candles=candles,
        oi_data=oi_aligned,
    )

    if args.save:
        dataset_name = f"{args.hypothesis_id}_{args.symbol}_{args.timeframe}"
        path = save_features(features, dataset_name)
        logger.info(f"Features saved to {path}")

    logger.info("Feature summary:")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Columns: {list(features.columns)}")
    logger.info(f"  Date range: {features.index.min()} -> {features.index.max()}")
    non_null = features.dropna()
    logger.info(f"  Complete rows: {len(non_null)} / {len(features)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
