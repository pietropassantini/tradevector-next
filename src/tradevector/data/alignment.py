"""Align external data to candle timestamps.

Rules (non-negotiable):
- No future data peeking.
- Use only info available at candle timestamp.
- Daily data cant be used as intraday.
- Event-based data must be explicitly aggregated.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AlignMethod = Literal["last_known", "asof", "nearest_prev"]


def infer_data_granularity(df: pd.DataFrame) -> Optional[str]:
    if len(df) < 2:
        return None
    median_seconds = df.index.to_series().diff().median().total_seconds()
    for label, seconds in [
        ("1m", 60), ("5m", 300), ("15m", 900), ("30m", 1800),
        ("1h", 3600), ("4h", 14400), ("8h", 28800), ("1d", 86400),
    ]:
        if abs(median_seconds - seconds) < (seconds * 0.1):
            return label
    return f"{median_seconds:.0f}s"


def align_to_candles(
    external_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    method: AlignMethod = "last_known",
    max_lookback: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    if len(external_df) == 0 or len(candles_df) == 0:
        return pd.DataFrame()
    if len(external_df.columns) == 0 or len(candles_df.columns) == 0:
        return pd.DataFrame()

    external_df = external_df[~external_df.index.duplicated(keep="last")]

    combined = candles_df[[]].copy()
    combined["_candle_ts"] = combined.index

    for col in external_df.columns:
        external_sorted = external_df[[col]].sort_index()

        if method == "last_known":
            forward_filled = external_sorted.reindex(combined.index, method="ffill")
            if max_lookback:
                ext_reindexed = external_sorted.reindex(combined.index)
                gap_mask = (combined.index - ext_reindexed.index.to_series().reindex(
                    combined.index, method="ffill"
                ).values) > max_lookback
                forward_filled.loc[gap_mask] = np.nan
            combined[col] = forward_filled[col]

        elif method == "asof":
            ext_values = [external_sorted[col].asof(ts) for ts in combined.index]
            combined[col] = ext_values

        elif method == "nearest_prev":
            indices = external_sorted.index.searchsorted(combined.index, side="right") - 1
            valid = (indices >= 0) & (indices < len(external_sorted))
            combined[col] = np.nan
            combined.loc[valid, col] = external_sorted[col].iloc[indices[valid]].values

    combined = combined.drop(columns=["_candle_ts"])
    return combined


def validate_alignment(aligned_df: pd.DataFrame) -> dict:
    report = {
        "rows": len(aligned_df),
        "total_nulls": int(aligned_df.isnull().sum().sum()),
        "nulls_per_column": {k: int(v) for k, v in aligned_df.isnull().sum().items() if v > 0},
        "coverage": {},
    }

    for col in aligned_df.columns:
        non_null = aligned_df[col].notna().sum()
        report["coverage"][col] = f"{non_null}/{len(aligned_df)} ({100*non_null/max(len(aligned_df),1):.1f}%)"

    return report


def align_and_validate(
    external_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    source_name: str = "unknown",
    output_dir: Optional[str] = None,
    method: AlignMethod = "last_known",
    max_lookback: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    ext_granularity = infer_data_granularity(external_df)
    candle_granularity = infer_data_granularity(candles_df)
    logger.info(f"External data granularity: {ext_granularity}")
    logger.info(f"Candle granularity: {candle_granularity}")

    aligned = align_to_candles(external_df, candles_df, method=method, max_lookback=max_lookback)

    validation = validate_alignment(aligned)
    logger.info(f"Aligned {validation['rows']} rows")
    logger.info(f"Total nulls: {validation['total_nulls']}")
    logger.info(f"Coverage: {validation['coverage']}")

    if output_dir:
        from pathlib import Path

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        parquet_path = out / f"{source_name}_aligned.parquet"
        aligned.to_parquet(parquet_path)
        logger.info(f"Saved aligned data to {parquet_path}")

    return aligned
