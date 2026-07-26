"""Random baseline for signal validation.

Each P0 must compare the signal against a random baseline with the same
trade frequency to verify the signal is not just noise.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_permuted_signals(
    signal: pd.Series,
    n_trials: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Baseline per permutazione: stessi valori del segnale, ordine rimescolato.

    Conserva esattamente la distribuzione del segnale, quindi le soglie sui
    quantili si comportano allo stesso modo e il confronto misura solo ciò che
    interessa: se l'istante in cui un valore si presenta porta informazione.

    Il baseline precedente generava una serie sparsa di -1/0/+1: con la maggior
    parte dei valori a zero il quantile 0.8 cadeva anch'esso su zero, quindi il
    "top quantile" finiva per contenere l'80-85% delle osservazioni e la sua
    gross expectancy era zero per costruzione, non per assenza di edge.
    """
    rng = np.random.default_rng(seed)
    valori = signal.dropna().to_numpy()
    out = pd.DataFrame(index=signal.index)
    for i in range(n_trials):
        permutati = np.full(len(signal), np.nan)
        posizioni = np.flatnonzero(signal.notna().to_numpy())
        permutati[posizioni] = rng.permutation(valori)
        out[f"random_{i}"] = permutati
    return out


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
    quantile_window: Optional[int] = 200,
) -> dict:
    from .signal_probe import compute_forward_returns, compute_signal_metrics

    prices = features["close"]
    signal = features[signal_column]
    forward_returns = compute_forward_returns(prices, horizons)

    # Segnale reale e permutazioni devono passare dalla stessa costruzione delle
    # soglie, altrimenti il confronto misura la differenza tra due metodi.
    actual_metrics = compute_signal_metrics(
        signal, forward_returns, horizons, quantile_window=quantile_window
    )

    random_signals = generate_permuted_signals(signal, n_trials, seed)

    random_metrics_list = []
    for col in random_signals.columns:
        r_metrics = compute_signal_metrics(
            random_signals[col], forward_returns, horizons, quantile_window=quantile_window
        )
        random_metrics_list.append(r_metrics)

    comparison = {}
    for h in horizons:
        comparison[f"h{h}"] = compare_to_random(actual_metrics, random_metrics_list, h)

    return {
        "actual_metrics": actual_metrics,
        "random_baseline": comparison,
    }
