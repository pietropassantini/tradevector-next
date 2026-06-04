"""Download historical Binance open interest data and save as parquet."""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# OI Data api endpoint (public, no API key required).
BINANCE_OI_URL = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"


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


def fetch_historical_oi(
    symbol: str,
    period: str = "5m",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> list[dict]:
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    all_data = []
    while True:
        try:
            resp = requests.get(BINANCE_OI_HIST_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            if len(data) < limit:
                break
            params["startTime"] = data[-1]["timestamp"] + 1
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"Failed to fetch historical OI for {symbol}: {e}")
            break

    logger.info(f"Fetched {len(all_data)} OI records for {symbol}")
    return all_data


def oi_hist_to_dataframe(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    return df


def download_open_interest(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    raw_dir: Optional[Path] = None,
) -> Path:
    if raw_dir is None:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "binance" / "open_interest" / symbol

    raw_dir.mkdir(parents=True, exist_ok=True)

    data = fetch_historical_oi(
        symbol=symbol,
        period=period,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )

    if not data:
        logger.warning(f"No data retrieved for {symbol}")
        return raw_dir

    raw_path = raw_dir / f"{symbol}_{period}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, default=str)

    df = oi_hist_to_dataframe(data)
    parquet_path = raw_dir / f"{symbol}_{period}.parquet"

    # Accumula: merge con storico esistente, no sovrascrittura
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info(f"Merged: {len(df)} rows total (was {len(existing)})")

    df.to_parquet(parquet_path)

    logger.info(f"Saved {len(df)} rows to {parquet_path}")
    logger.info(f"Date range: {df.index.min()} -> {df.index.max()}")
    logger.info(f"Raw response saved to {raw_path}")

    return raw_dir


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
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("timestamp").sort_index()

    parquet_path = raw_dir / f"{symbol}_{interval}.parquet"

    # Accumula: merge con storico esistente, no sovrascrittura
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info(f"Merged: {len(df)} klines total (was {len(existing)})")

    df.to_parquet(parquet_path)
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

    # Binance openInterestHist returns the *newest* records when no startTime is
    # set, so forward pagination stalls immediately. Anchor startTime in the past
    # to walk forward across the full retained window.
    oi_start_time = args.start_time
    if oi_start_time is None:
        oi_start_time = int(
            (datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000
        )

    download_open_interest(
        symbol=args.symbol,
        period=args.period,
        limit=args.limit,
        start_time=oi_start_time,
        end_time=args.end_time,
    )

    if args.download_klines:
        download_klines(
            symbol=args.symbol,
            interval=args.klines_interval,
            start_time=args.start_time,
            end_time=args.end_time,
        )


if __name__ == "__main__":
    main()
