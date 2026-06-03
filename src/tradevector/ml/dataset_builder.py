"""Dataset builder for supervised ML phase.

Converts features + forward returns into supervised learning datasets
with temporal integrity (no future leakage).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_forward_returns(
    df: pd.DataFrame,
    horizons: list[int],
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    for h in horizons:
        result[f"fwd_return_{h}"] = result[price_col].shift(-h) / result[price_col] - 1
    return result


def add_labels(
    df: pd.DataFrame,
    horizon: int,
    target_type: str = "directional",
    threshold: float = 0.005,
    atr_col: Optional[str] = "atr",
    k: float = 2.0,
) -> pd.DataFrame:
    result = df.copy()
    fwd_col = f"fwd_return_{horizon}"

    if target_type == "directional":
        result["target"] = np.where(
            result[fwd_col] > threshold, 1,
            np.where(result[fwd_col] < -threshold, -1, 0)
        )
    elif target_type == "volatility_expansion":
        if atr_col and atr_col in result.columns:
            norm_fwd = result[fwd_col].abs() / result[atr_col]
            result["target"] = np.where(norm_fwd > k, 1, 0)
        else:
            result["target"] = np.where(result[fwd_col].abs() > threshold, 1, 0)
    elif target_type == "breakout":
        result["target"] = np.where(result[fwd_col].abs() > threshold, 1, 0)
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    return result


def apply_embargo(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    return df.iloc[:-horizon] if horizon > 0 else df


def build_supervised_dataset(
    features_df: pd.DataFrame,
    horizon: int,
    target_type: str = "directional",
    threshold: float = 0.005,
    exclude_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = ["close", "volume", "returns", "log_returns"]

    df = features_df.copy()

    df = add_forward_returns(df, [horizon])
    df = add_labels(df, horizon, target_type, threshold)
    df = apply_embargo(df, horizon)

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not c.startswith("fwd_return_")
                    and c != "target"]
    df = df.dropna(subset=feature_cols + ["target"])

    logger.info(f"Built dataset: {len(df)} samples, {len(feature_cols)} features")
    logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

    return df
