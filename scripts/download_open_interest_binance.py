"""Download historical Binance open interest data and save as parquet."""

import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Il collector generico vive in libreria: fetch con fallback sullo startTime
# fuori retention, ancoraggio all'archivio e merge non distruttivo sono gli
# stessi per tutte le serie futures/data, e una sola copia evita che divergano.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.data.binance_futures import (  # noqa: E402
    accumulate,
    coerce_numeric,
    download_metric,
)

BINANCE_OI_URL = "https://fapi.binance.com/fapi/v1/openInterest"


def load_data_sources() -> dict:
    config_path = PROJECT_ROOT / "config" / "data_sources.yml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_current_oi(symbol: str) -> Optional[dict]:
    params = {"symbol": symbol}
    try:
        resp = requests.get(BINANCE_OI_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        data["fetch_timestamp"] = datetime.now(timezone.utc).isoformat()
        return data
    except Exception as e:
        logger.error(f"Failed to fetch OI for {symbol}: {e}")
        return None


def download_open_interest(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    raw_dir: Optional[Path] = None,
    days: int = 29,
) -> bool:
    if start_time is not None or end_time is not None:
        logger.warning("start-time/end-time espliciti non supportati, uso l'archivio")
    return download_metric(
        metric="open_interest", symbol=symbol, period=period,
        limit=limit, days=days, save_raw_json=True,
    )


def download_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    raw_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "binance" / "klines" / symbol

    raw_dir.mkdir(parents=True, exist_ok=True)

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    all_data = []
    while True:
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            if len(data) < limit:
                break
            params["startTime"] = data[-1][0] + 1
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            break

    if not all_data:
        logger.warning(f"No klines data retrieved for {symbol}")
        return pd.DataFrame()

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(all_data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    # Tutte le colonne a numerico, non solo OHLCV: lasciarne alcune stringa
    # produce un concat misto con l'archivio che parquet rifiuta di scrivere.
    df = coerce_numeric(df)

    parquet_path = raw_dir / f"{symbol}_{interval}.parquet"
    df = accumulate(df, parquet_path)

    logger.info(f"Saved {len(df)} klines to {parquet_path}")
    logger.info(f"Date range: {df.index.min()} -> {df.index.max()}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download Binance Open Interest data")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--period", default="5m", help="Data period (default: 5m)")
    parser.add_argument("--limit", type=int, default=500, help="Records per request (default: 500)")
    parser.add_argument("--start-time", type=int, default=None, help="Start timestamp (ms)")
    parser.add_argument("--end-time", type=int, default=None, help="End timestamp (ms)")
    parser.add_argument("--days", type=int, default=30,
                        help="Days of history to fetch when --start-time omitted "
                             "(Binance openInterestHist keeps ~30d; default: 30)")
    parser.add_argument("--download-klines", action="store_true", help="Download klines too")
    parser.add_argument("--klines-interval", default="1h", help="Klines interval (default: 1h)")

    args = parser.parse_args()

    # Con --start-time assente lo startTime viene ancorato all'ultimo timestamp
    # già in archivio (vedi resolve_start_time), non a una finestra fissa: a 30
    # giorni esatti l'endpoint risponde 400 e il download fallisce in silenzio.
    ok = download_open_interest(
        symbol=args.symbol,
        period=args.period,
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
        days=args.days,
    )

    if args.download_klines:
        klines = download_klines(
            symbol=args.symbol,
            interval=args.klines_interval,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        if len(klines) == 0:
            ok = False

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
