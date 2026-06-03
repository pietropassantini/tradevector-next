"""Feature builder for Open Interest Compression Breakout hypothesis.

Features are kept simple by design — no tuning, no post-hoc selection.
All features must be computable without future data (no temporal leakage).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_oi_features(
    candles: pd.DataFrame,
    oi_data: pd.DataFrame,
    funding_data: Optional[pd.DataFrame] = None,
    lookback_periods: int = 20,
    zscore_periods: int = 100,
) -> pd.DataFrame:
    features = pd.DataFrame(index=candles.index)
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(candles.columns):
        missing = required_cols - set(candles.columns)
        raise ValueError(f"Candles missing required columns: {missing}")

    oi_aligned = _align_oi_to_candles(oi_data, candles)

    features["close"] = candles["close"]
    features["volume"] = candles["volume"]

    features["returns"] = candles["close"].pct_change(fill_method=None)
    features["log_returns"] = np.log(candles["close"] / candles["close"].shift(1))

    features["high_low_range"] = candles["high"] - candles["low"]
    features["price_range_norm"] = (
        (candles["high"].rolling(lookback_periods).max()
         - candles["low"].rolling(lookback_periods).min())
        / candles["close"]
    )
    atr = _compute_atr(candles, lookback_periods)
    features["atr"] = atr
    features["atr_norm"] = atr / candles["close"]
    features["volume_ratio"] = candles["volume"] / candles["volume"].rolling(lookback_periods).mean()
    features["volume_zscore"] = (
        (candles["volume"] - candles["volume"].rolling(zscore_periods).mean())
        / candles["volume"].rolling(zscore_periods).std()
    )

    features["oi"] = oi_aligned.get("sumOpenInterest", np.nan)
    features["oi_change"] = features["oi"].diff()
    features["oi_change_pct"] = features["oi"].pct_change(fill_method=None)
    features["oi_zscore"] = (
        (features["oi"] - features["oi"].rolling(zscore_periods).mean())
        / features["oi"].rolling(zscore_periods).std()
    )
    features["oi_ma"] = features["oi"].rolling(lookback_periods).mean()
    features["oi_ma_ratio"] = features["oi"] / features["oi_ma"]

    features["oi_acceleration"] = features["oi_change_pct"].diff()

    features["oi_price_divergence"] = _compute_divergence(
        features["oi"], candles["close"], lookback_periods
    )

    if funding_data is not None and "fundingRate" in funding_data.columns:
        funding_aligned = funding_data["fundingRate"].reindex(candles.index, method="ffill")
        features["funding_rate"] = pd.to_numeric(funding_aligned, errors="coerce")
        features["funding_zscore"] = (
            (features["funding_rate"] - features["funding_rate"].rolling(zscore_periods).mean())
            / features["funding_rate"].rolling(zscore_periods).std()
        )
        features["funding_context"] = np.sign(features["funding_rate"])

    features = _add_lagged_features(features, ["oi_change_pct", "oi_zscore"], lags=[1, 2, 3])

    features = features.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Built OI features: {len(features)} rows x {len(features.columns)} cols")
    logger.info(f"Date range: {features.index.min()} -> {features.index.max()}")
    logger.info(f"Null fraction: {features.isnull().mean().mean():.2%}")

    return features


def _align_oi_to_candles(
    oi_data: pd.DataFrame,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    oi_cols = oi_data.select_dtypes(include=[np.number]).columns
    if len(oi_cols) == 0:
        return pd.DataFrame(index=candles.index)
    aligned = pd.DataFrame(index=candles.index)
    for col in oi_cols:
        aligned[col] = oi_data[col].reindex(candles.index, method="ffill")
    return aligned


def _compute_atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = candles["high"], candles["low"], candles["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _compute_divergence(
    oi: pd.Series,
    price: pd.Series,
    lookback: int,
) -> pd.Series:
    oi_change = oi.pct_change(lookback, fill_method=None)
    price_change = price.pct_change(lookback, fill_method=None)
    divergence = np.where(
        (oi_change > 0) & (price_change < 0), 1,
        np.where((oi_change < 0) & (price_change > 0), -1, 0)
    )
    return pd.Series(divergence, index=oi.index)


def _add_lagged_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def save_features(
    features: pd.DataFrame,
    dataset_name: str,
    output_dir: Optional[str] = None,
) -> str:
    from pathlib import Path

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[3] / "data" / "features"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parquet_path = out / f"{dataset_name}.parquet"
    features.to_parquet(parquet_path)
    logger.info(f"Saved features to {parquet_path}")
    return str(parquet_path)
