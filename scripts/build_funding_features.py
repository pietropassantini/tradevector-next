"""Build funding features dallo storico e salva come parquet."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.features.funding_features import (
    load_storico_klines,
    load_storico_funding,
    build_funding_features,
    save_funding_features,
)


def main():
    parser = argparse.ArgumentParser(description="Build funding features dallo storico")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--hypothesis-id", default="funding_positioning_reversal")
    parser.add_argument("--zscore-periods", type=int, default=90,
                        help="Finestra zscore in record funding 8h (default=90 = 30gg)")
    args = parser.parse_args()

    logger.info(f"Caricamento klines {args.symbol} {args.timeframe}...")
    candles = load_storico_klines(args.symbol, args.timeframe)

    logger.info(f"Caricamento funding {args.symbol}...")
    funding = load_storico_funding(args.symbol)

    logger.info("Building features...")
    features = build_funding_features(
        candles=candles,
        funding=funding,
        zscore_periods=args.zscore_periods,
    )

    dataset_name = f"{args.hypothesis_id}_{args.symbol}_{args.timeframe}"
    path = save_funding_features(features, dataset_name)
    logger.info(f"Features salvate: {path}")
    logger.info(f"Shape: {features.shape}")
    logger.info(f"Righe complete: {len(features.dropna())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
