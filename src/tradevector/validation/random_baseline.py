"""Random baseline for signal validation.

Each P0 must compare the signal against a random baseline with the same
trade frequency to verify the signal is not just noise.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_random_signals(
    index: pd.DatetimeIndex,
    signal_frequency: float,
    n_trials: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    n_signals = int(n * signal_frequency)
    random_signals = pd.DataFrame(index=index)
    for i in range(n_trials):
        positions = np.zeros(n)
        signal_idx = rng.choice(n, size=max(n_signals, 1), replace=False)
        directions = rng.choice([1, -1], size=len(signal_idx))
        positions[signal_idx] = directions
        random_signals[f"random_{i}"] = positions
    return random_signals


def compare_to_random(
    signal_metrics: dict,
    random_metrics: list[dict],
    horizon: int,
) -> dict:
    h_key = f"h{horizon}"
    actual = signal_metrics.get(h_key, {})

    if "error" in actual:
        return {"error": "actual signal metrics not available"}

    random_ge = [r.get(h_key, {}).get("gross_expectancy", 0) for r in random_metrics]
    random_ge = [v for v in random_ge if isinstance(v, (int, float))]

    if not random_ge:
        return {"error": "no random baseline data"}

    actual_ge = actual.get("gross_expectancy", 0)
    random_ge_mean = np.mean(random_ge)
    random_ge_std = np.std(random_ge)
    pct_better = np.mean([1 if actual_ge > r else 0 for r in random_ge])

    return {
        "actual_gross_expectancy": actual_ge,
        "random_mean_gross_expectancy": float(random_ge_mean),
        "random_std_gross_expectancy": float(random_ge_std),
        "pct_random_worse": float(pct_better),
        "z_score_vs_random": float((actual_ge - random_ge_mean) / random_ge_std)
        if random_ge_std > 0 else 0,
        "n_random_trials": len(random_ge),
    }


def run_random_baseline(
    features: pd.DataFrame,
    signal_column: str,
    horizons: list[int],
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    from .signal_probe import compute_forward_returns, compute_signal_metrics

    prices = features["close"]
    signal = features[signal_column]
    forward_returns = compute_forward_returns(prices, horizons)

    actual_metrics = compute_signal_metrics(signal, forward_returns, horizons)

    signal_frequency = (signal.abs() > signal.std()).mean() if signal.std() > 0 else 0.1
    random_signals = generate_random_signals(features.index, signal_frequency, n_trials, seed)

    random_metrics_list = []
    for col in random_signals.columns:
        r_metrics = compute_signal_metrics(random_signals[col], forward_returns, horizons)
        random_metrics_list.append(r_metrics)

    comparison = {}
    for h in horizons:
        comparison[f"h{h}"] = compare_to_random(actual_metrics, random_metrics_list, h)

    return {
        "actual_metrics": actual_metrics,
        "random_baseline": comparison,
    }
