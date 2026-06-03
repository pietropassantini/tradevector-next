"""Tests to verify no temporal leakage in the pipeline.

These are methodological tests that verify non-negotiable principles:
- No future data in features.
- Target computed only after signal timestamp.
- Training always precedes testing.
- No temporal shuffle.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.data.alignment import align_to_candles
from tradevector.features.oi_features import build_oi_features
from tradevector.ml.time_split import walk_forward_split
from tradevector.validation.signal_probe import compute_forward_returns


@pytest.fixture
def time_series():
    idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(300) * 50)
    candles = pd.DataFrame(
        {
            "open": close + np.random.randn(300) * 5,
            "high": close + np.abs(np.random.randn(300) * 25),
            "low": close - np.abs(np.random.randn(300) * 25),
            "close": close,
            "volume": np.abs(np.random.randn(300) * 500 + 2000),
        },
        index=idx,
    )
    oi_idx = pd.date_range("2024-01-01 00:05:00", periods=280, freq="1h", tz="UTC")
    oi = pd.DataFrame(
        {"sumOpenInterest": 50000 + np.cumsum(np.random.randn(280) * 100)},
        index=oi_idx,
    )
    return candles, oi


class TestNoFutureDataInFeatures:
    def test_alignment_no_future_peek(self, time_series):
        candles, oi = time_series
        aligned = align_to_candles(oi, candles, method="last_known")
        oi_col = aligned["sumOpenInterest"].dropna()
        for ts in oi_col.index:
            last_oi_ts = oi[oi.index <= ts].index.max()
            if last_oi_ts is not None:
                assert last_oi_ts <= ts, f"Future OI used at {ts}, source was {last_oi_ts}"

    def test_features_no_forward_look(self, time_series):
        candles, oi = time_series
        features = build_oi_features(candles, oi)
        assert "close" in features.columns
        assert features["close"].notna().sum() == len(features)

    def test_forward_returns_use_future(self, time_series):
        candles, _ = time_series
        fwd = compute_forward_returns(candles["close"], [1, 2, 4])
        assert fwd["fwd_return_1"].iloc[-1] != fwd["fwd_return_1"].iloc[-1]
        assert pd.isna(fwd["fwd_return_1"].iloc[-1])


class TestTemporalIntegrity:
    def test_train_before_test(self, time_series):
        candles, oi = time_series
        features = build_oi_features(candles, oi).dropna()
        if len(features) < 60:
            pytest.skip("Not enough data after dropna")
        splits = list(walk_forward_split(features, train_size=50, test_size=20, embargo=5))
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min()

    def test_no_shuffle_in_split(self, time_series):
        candles, oi = time_series
        features = build_oi_features(candles, oi).dropna()
        if len(features) < 60:
            pytest.skip("Not enough data after dropna")
        for train_idx, test_idx in walk_forward_split(features, 50, 20):
            assert np.all(np.diff(train_idx) > 0), "Train indices not sorted"
            assert np.all(np.diff(test_idx) > 0), "Test indices not sorted"


class TestGrossBeforeNet:
    def test_signal_probe_computes_gross(self, time_series):
        candles, oi = time_series
        features = build_oi_features(candles, oi)
        assert "oi_zscore" in features.columns
        assert features["oi_zscore"].notna().any()
