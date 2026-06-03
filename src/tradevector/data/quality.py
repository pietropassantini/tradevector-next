"""Data quality checking module."""

import numpy as np
import pandas as pd


def check_duplicates(df: pd.DataFrame) -> dict:
    dupes = df.index.duplicated().sum()
    return {"duplicate_count": int(dupes), "has_duplicates": dupes > 0}


def check_gaps(df: pd.DataFrame, expected_interval_seconds: int = None) -> dict:
    if len(df) < 2:
        return {"gaps": 0, "max_gap": None}
    diffs = df.index.to_series().diff().dropna()
    if expected_interval_seconds:
        median = pd.Timedelta(seconds=expected_interval_seconds)
    else:
        median = diffs.median()
    threshold = median * 3
    gaps = diffs[diffs > threshold]
    return {
        "gap_count": len(gaps),
        "max_gap_seconds": float(gaps.max().total_seconds()) if len(gaps) > 0 else 0,
    }


def check_nulls(df: pd.DataFrame) -> dict:
    nulls = df.isnull().sum()
    return {
        "total_nulls": int(nulls.sum()),
        "per_column": {k: int(v) for k, v in nulls.items()},
    }


def check_outliers(df: pd.DataFrame, sigma: float = 5.0) -> dict:
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        z = np.abs((col_data - col_data.mean()) / col_data.std())
        result[col] = int((z > sigma).sum())
    return result
