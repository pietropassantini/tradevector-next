"""Data loaders for common data formats."""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.index.name != "timestamp" and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    # Guard difensiva: timestamp duplicati (es. ribar corrente riscaricato)
    # romperebbero allineamento e quantili a valle. Tieni l'ultima osservazione.
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_oi_data(
    symbol: str = "BTCUSDT",
    period: str = "5m",
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if data_dir is None:
        from pathlib import Path as _Path

        data_dir = _Path(__file__).resolve().parents[3] / "data"

    path = data_dir / "raw" / "binance" / "open_interest" / symbol / f"{symbol}_{period}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"OI data not found: {path}")
    return load_parquet(path)


def load_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if data_dir is None:
        from pathlib import Path as _Path

        data_dir = _Path(__file__).resolve().parents[3] / "data"

    path = data_dir / "raw" / "binance" / "klines" / symbol / f"{symbol}_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Klines not found: {path}")
    return load_parquet(path)


def load_features(
    dataset_name: str,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if data_dir is None:
        from pathlib import Path as _Path

        data_dir = _Path(__file__).resolve().parents[3] / "data"

    path = data_dir / "features" / f"{dataset_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    return load_parquet(path)
