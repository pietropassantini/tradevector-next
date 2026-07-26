"""Accumula le serie futures/data di Binance (retention ~21 giorni).

Va lanciato ogni ora e indipendentemente dalle strategie: l'archivio è l'unico
posto dove queste serie sopravvivono oltre le tre settimane, e nessuno le vende
a posteriori. Esce non-zero se anche una sola serie fallisce, così il fallimento
è visibile in systemd invece di restare silenzioso.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.data.binance_futures import FUTURES_DATA_ENDPOINTS, download_metric

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_METRICS = list(FUTURES_DATA_ENDPOINTS)


def main():
    parser = argparse.ArgumentParser(description="Collector serie futures/data Binance")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help=f"Lista separata da virgole (default: {','.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                        help=f"Serie da raccogliere (default: tutte — {','.join(DEFAULT_METRICS)})")
    parser.add_argument("--period", default="1h", help="Granularità (default: 1h)")
    parser.add_argument("--days", type=int, default=29,
                        help="Lookback massimo su archivio vuoto (default: 29)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    sconosciute = [m for m in metrics if m not in FUTURES_DATA_ENDPOINTS]
    if sconosciute:
        logger.error(f"Metriche sconosciute: {sconosciute}")
        logger.error(f"Disponibili: {list(FUTURES_DATA_ENDPOINTS)}")
        return 2

    falliti = []
    for symbol in symbols:
        for metric in metrics:
            if not download_metric(
                metric=metric, symbol=symbol, period=args.period, days=args.days
            ):
                falliti.append(f"{metric}/{symbol}")

    totali = len(symbols) * len(metrics)
    if falliti:
        logger.error(f"Falliti {len(falliti)}/{totali}: {', '.join(falliti)}")
        return 1

    logger.info(f"Raccolte {totali}/{totali} serie")
    return 0


if __name__ == "__main__":
    sys.exit(main())
