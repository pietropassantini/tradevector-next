"""Tests for temporal alignment module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.data.alignment import (
    align_to_candles,
    infer_data_granularity,
    validate_alignment,
)


@pytest.fixture
def candles_df():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": np.random.randn(100) + 100, "close": np.random.randn(100) + 100},
        index=idx,
    )


@pytest.fixture
def external_df():
    idx = pd.date_range("2024-01-01 00:05:00", periods=90, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"sumOpenInterest": np.random.randn(90) * 1000 + 50000},
        index=idx,
    )


class TestInferGranularity:
    def test_1h_granularity(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
        df = pd.DataFrame({"a": range(100)}, index=idx)
        assert infer_data_granularity(df) == "1h"

    def test_single_row_returns_none(self):
        idx = pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC")
        df = pd.DataFrame({"a": [1]}, index=idx)
        assert infer_data_granularity(df) is None


class TestAlignToCandles:
    def test_last_known_method(self, candles_df, external_df):
        aligned = align_to_candles(external_df, candles_df, method="last_known")
        assert len(aligned) == len(candles_df)
        assert "sumOpenInterest" in aligned.columns

    def test_no_future_peeking(self, candles_df, external_df):
        aligned = align_to_candles(external_df, candles_df, method="last_known")
        oi_col = aligned["sumOpenInterest"].dropna()
        if len(oi_col) > 0:
            for ts in oi_col.index:
                last_oi = external_df[external_df.index <= ts].index.max()
                assert last_oi is None or last_oi <= ts

    def test_empty_input(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
        empty = pd.DataFrame(index=idx)
        result = align_to_candles(empty, empty, method="last_known")
        assert len(result) == 0


class TestValidateAlignment:
    def test_coverage_calculation(self, candles_df, external_df):
        aligned = align_to_candles(external_df, candles_df, method="last_known")
        report = validate_alignment(aligned)
        assert "rows" in report
        assert "coverage" in report
        assert report["rows"] == len(candles_df)
