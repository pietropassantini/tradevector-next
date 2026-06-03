"""Decile analysis for ML signal probe.

For each score decile: sample count, mean/median forward return,
gross expectancy, hit rate, tail move frequency, baseline comparison.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def decile_analysis(
    scores: pd.Series,
    forward_returns: pd.Series,
    n_deciles: int = 10,
) -> dict:
    aligned = pd.DataFrame({"score": scores, "fwd_return": forward_returns}).dropna()
    if len(aligned) < n_deciles * 2:
        return {"error": "insufficient data for decile analysis", "samples": len(aligned)}

    aligned["decile"] = pd.qcut(aligned["score"], n_deciles, labels=False, duplicates="drop")

    results = {}
    for d in range(n_deciles):
        mask = aligned["decile"] == d
        if mask.sum() == 0:
            continue
        d_returns = aligned.loc[mask, "fwd_return"]
        results[f"decile_{d+1}"] = {
            "count": int(mask.sum()),
            "mean_forward_return": float(d_returns.mean()),
            "median_forward_return": float(d_returns.median()),
            "gross_expectancy": float(d_returns.mean()),
            "hit_rate": float((d_returns > 0).mean()),
            "tail_5pct": float(d_returns.quantile(0.05)),
            "tail_95pct": float(d_returns.quantile(0.95)),
            "std": float(d_returns.std()),
        }

    top_decile = results.get(f"decile_{n_deciles}", {})
    bottom_decile = results.get("decile_1", {})
    monotonicity = _check_monotonicity(results)

    return {
        "deciles": results,
        "top_decile_expectancy": top_decile.get("gross_expectancy", 0),
        "bottom_decile_expectancy": bottom_decile.get("gross_expectancy", 0),
        "score_monotonicity": monotonicity,
        "total_samples": len(aligned),
    }


def _check_monotonicity(decile_results: dict) -> float:
    means = []
    for i in range(1, 11):
        key = f"decile_{i}"
        if key in decile_results:
            means.append(decile_results[key].get("mean_forward_return", 0))
    if len(means) < 2:
        return 0.0
    diffs = np.diff(means)
    monotonic = np.mean(diffs > 0) if np.mean(np.abs(diffs)) > 0 else 0.0
    return float(monotonic)
