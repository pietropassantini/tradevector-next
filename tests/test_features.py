"""Tests for OI feature builder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.features.oi_features import build_oi_features


@pytest.fixture
def sample_candles():
    idx = pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC")
    np.random.seed(42)
    close = 50000 + np.cumsum(np.random.randn(500) * 100)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(500) * 10,
            "high": close + np.abs(np.random.randn(500) * 50),
            "low": close - np.abs(np.random.randn(500) * 50),
            "close": close,
            "volume": np.random.randn(500) * 100 + 1000,
        },
        index=idx,
    )


@pytest.fixture
def sample_oi():
    idx = pd.date_range("2024-01-01 00:05:00", periods=450, freq="1h", tz="UTC")
    np.random.seed(43)
    return pd.DataFrame(
        {"sumOpenInterest": 50000 + np.cumsum(np.random.randn(450) * 200)},
        index=idx,
    )


class TestOICompressionBreakout:
    def _get_features(self, candles, oi):
        return build_oi_features(candles, oi)

    def test_output_shape(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        assert len(features) == len(sample_candles)

    def test_oi_features_present(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        expected = [
            "oi_change_pct", "oi_zscore", "price_range_norm",
            "atr_norm", "oi_price_divergence",
        ]
        for col in expected:
            assert col in features.columns, f"Missing column: {col}"

    def test_oi_change_computed(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        assert features["oi_change_pct"].notna().sum() > 0

    def test_price_range_norm_bounded(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        prn = features["price_range_norm"].dropna()
        assert (prn >= 0).all()

    def test_atr_positive(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        atr = features["atr"].dropna()
        assert (atr > 0).all()

    def test_divergence_values(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        div = features["oi_price_divergence"].dropna()
        assert div.isin([-1, 0, 1]).all()

    def test_no_inf_values(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        numeric = features.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_lags_present(self, sample_candles, sample_oi):
        features = self._get_features(sample_candles, sample_oi)
        assert "oi_change_pct_lag1" in features.columns
        assert "oi_zscore_lag1" in features.columns
