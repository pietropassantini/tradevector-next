"""Tests for temporal split module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.ml.time_split import (
    walk_forward_split,
    expanding_window_split,
)


@pytest.fixture
def sample_df():
    idx = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    return pd.DataFrame({"a": range(100)}, index=idx)


class TestWalkForwardSplit:
    def test_basic_split(self, sample_df):
        splits = list(walk_forward_split(sample_df, train_size=50, test_size=20))
        assert len(splits) > 0

    def test_training_before_testing(self, sample_df):
        for train_idx, test_idx in walk_forward_split(sample_df, 50, 20, embargo=0):
            assert train_idx.max() < test_idx.min()

    def test_embargo_creates_gap(self, sample_df):
        train_idx, test_idx = next(walk_forward_split(sample_df, 50, 20, embargo=5))
        gap = test_idx.min() - train_idx.max()
        assert gap >= 5

    def test_no_splits_when_data_too_small(self, sample_df):
        small = sample_df.iloc[:10]
        splits = list(walk_forward_split(small, 50, 20))
        assert len(splits) == 0


class TestExpandingWindowSplit:
    def test_basic_expanding(self, sample_df):
        splits = list(expanding_window_split(sample_df, min_train_size=40, test_size=20))
        assert len(splits) > 0

    def test_train_grows(self, sample_df):
        prev_len = 0
        for train_idx, _ in expanding_window_split(sample_df, 40, 20):
            assert len(train_idx) >= prev_len
            prev_len = len(train_idx)

    def test_no_overlap_train_test(self, sample_df):
        for train_idx, test_idx in expanding_window_split(sample_df, 40, 20, embargo=5):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0
