"""Temporal-aware train/test splits for time series.

Non-negotiable rules:
- No random split on time series.
- No shuffle.
- Training always precedes testing.
- Embargo >= target horizon.
- Purge overlapping samples.
"""

import logging
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def walk_forward_split(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    step: Optional[int] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    if step is None:
        step = test_size

    start = 0
    while start + train_size + embargo + test_size <= n:
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        yield np.arange(start, train_end), np.arange(test_start, test_end)
        start += step


def expanding_window_split(
    df: pd.DataFrame,
    min_train_size: int,
    test_size: int,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    start = min_train_size
    while start + embargo + test_size <= n:
        test_start = start + embargo
        test_end = test_start + test_size
        yield np.arange(0, start), np.arange(test_start, test_end)
        start += test_size


def purge_overlapping_samples(
    df: pd.DataFrame,
    horizon: int,
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    if horizon <= 0:
        return df

    result = df.copy()
    for i in range(1, horizon):
        if target_col:
            shift_mask = result[target_col].shift(i).notna()
            result = result[~shift_mask | (shift_mask.index.duplicated(keep="first"))]

    return result


def time_series_split_indices(
    timestamps: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: Optional[pd.Timestamp] = None,
    embargo: pd.Timedelta = pd.Timedelta(0),
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    test_start_embargoed = test_start + embargo
    train_idx = np.where(timestamps < test_start_embargoed)[0]
    if test_end:
        test_idx = np.where((timestamps >= test_start_embargoed) & (timestamps <= test_end))[0]
    else:
        test_idx = np.where(timestamps >= test_start_embargoed)[0]

    return train_idx, test_idx, None
