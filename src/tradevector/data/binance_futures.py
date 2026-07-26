"""Collector per la famiglia futures/data di Binance.

Queste serie (open interest, posizionamento account, flusso taker) hanno una
retention di ~21 giorni: 500 record a granularità 1h. Non sono acquistabili a
posteriori, quindi l'unico modo di ottenere una finestra storica utilizzabile è
accumularle in locale a partire da oggi.

L'archivio deve sopravvivere alle strategie: la raccolta gira per conto suo, non
dentro lo scheduler di una strategia che può essere fermata o messa in dry-run.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"

# nome logico -> (path endpoint, sottocartella in data/raw/binance)
FUTURES_DATA_ENDPOINTS = {
    "open_interest": ("/futures/data/openInterestHist", "open_interest"),
    "top_long_short_account": (
        "/futures/data/topLongShortAccountRatio", "top_long_short_account"
    ),
    "top_long_short_position": (
        "/futures/data/topLongShortPositionRatio", "top_long_short_position"
    ),
    "global_long_short_account": (
        "/futures/data/globalLongShortAccountRatio", "global_long_short_account"
    ),
    "taker_long_short": ("/futures/data/takerlongshortRatio", "taker_long_short"),
}

# A 30 giorni esatti l'endpoint risponde 400 (-1130, "startTime is invalid"):
# è il bordo della retention. A 29 passa.
MAX_LOOKBACK_DAYS = 29


def metric_path(metric: str, symbol: str, period: str, raw_dir: Optional[Path] = None) -> Path:
    if metric not in FUTURES_DATA_ENDPOINTS:
        raise ValueError(f"Metrica sconosciuta: {metric}")
    _, folder = FUTURES_DATA_ENDPOINTS[metric]
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "binance"
    return raw_dir / folder / symbol / f"{symbol}_{period}.parquet"


def resolve_start_time(parquet_path: Path, days: int = MAX_LOOKBACK_DAYS) -> int:
    """Riparte dall'ultimo timestamp in archivio, senza superare la retention."""
    floor_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=min(days, MAX_LOOKBACK_DAYS))).timestamp()
        * 1000
    )
    if not parquet_path.exists():
        return floor_ms
    existing = pd.read_parquet(parquet_path)
    if len(existing) == 0:
        return floor_ms
    return max(int(existing.index.max().timestamp() * 1000) + 1, floor_ms)


def fetch_futures_data(
    endpoint: str,
    symbol: str,
    period: str = "1h",
    limit: int = 500,
    start_time: Optional[int] = None,
) -> list[dict]:
    params = {"symbol": symbol, "period": period, "limit": limit}
    if start_time:
        params["startTime"] = start_time

    url = BINANCE_FAPI + endpoint
    all_data: list[dict] = []
    dropped_start_time = False
    while True:
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 400 and "startTime" in params and not dropped_start_time:
                logger.warning(
                    f"{endpoint}: startTime rifiutato ({resp.text[:80]}), "
                    "riprovo senza — arriverà solo la finestra ancora in retention"
                )
                params.pop("startTime")
                dropped_start_time = True
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            if len(data) < limit:
                break
            params["startTime"] = data[-1]["timestamp"] + 1
            dropped_start_time = True
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"{endpoint} {symbol}: fetch fallito: {e}")
            break

    return all_data


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Tutte le colonne di valore a numerico, `symbol` resta stringa.

    Serve anche sull'archivio già a disco: gli storici scritti prima tenevano
    alcune colonne come stringa, e un concat tra object e float produce una
    colonna mista che pyarrow rifiuta di serializzare.
    """
    df = df.copy()
    for col in df.columns:
        if col == "symbol":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_dataframe(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return coerce_numeric(df)


def accumulate(df: pd.DataFrame, parquet_path: Path) -> pd.DataFrame:
    """Merge non distruttivo con l'archivio esistente."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if parquet_path.exists():
        existing = coerce_numeric(pd.read_parquet(parquet_path))
        before = len(existing)
        df = pd.concat([existing, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info(f"  merge: {before} -> {len(df)} righe")
    df.to_parquet(parquet_path)
    return df


def download_metric(
    metric: str,
    symbol: str,
    period: str = "1h",
    limit: int = 500,
    days: int = MAX_LOOKBACK_DAYS,
    raw_dir: Optional[Path] = None,
    save_raw_json: bool = False,
) -> bool:
    endpoint, _ = FUTURES_DATA_ENDPOINTS[metric]
    path = metric_path(metric, symbol, period, raw_dir)
    start_time = resolve_start_time(path, days)

    data = fetch_futures_data(
        endpoint=endpoint, symbol=symbol, period=period,
        limit=limit, start_time=start_time,
    )
    if not data:
        logger.error(f"{metric} {symbol}: nessun dato recuperato — FALLITO")
        return False

    if save_raw_json:
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / f"{symbol}_{period}_raw.json").write_text(
            json.dumps(data, default=str), encoding="utf-8"
        )

    merged = accumulate(to_dataframe(data), path)
    logger.info(
        f"{metric} {symbol}: {len(data)} nuovi | archivio {len(merged)} righe | "
        f"{merged.index.min()} -> {merged.index.max()}"
    )
    return True
