"""P0 Statistical Signal Probe — core metrics and analysis.

Measures gross edge before any strategy or cost model.
The key rule: mean gross is decisive, median gross is diagnostic.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_forward_returns(
    prices: pd.Series,
    horizons: list[int],
) -> pd.DataFrame:
    fwd = pd.DataFrame(index=prices.index)
    for h in horizons:
        fwd[f"fwd_return_{h}"] = prices.shift(-h) / prices - 1
        fwd[f"fwd_log_return_{h}"] = np.log(prices.shift(-h) / prices)
    return fwd


def compute_signal_metrics(
    signal: pd.Series,
    forward_returns: pd.DataFrame,
    horizons: list[int],
    top_quantile: float = 0.8,
    bottom_quantile: float = 0.2,
) -> dict:
    results = {}

    for h in horizons:
        fwd_col = f"fwd_return_{h}"
        if fwd_col not in forward_returns.columns:
            continue

        fwd = forward_returns[fwd_col]
        aligned = pd.DataFrame({"signal": signal, "fwd_return": fwd}).dropna()
        if len(aligned) < 30:
            results[f"h{h}"] = {"error": "insufficient data", "sample_size": len(aligned)}
            continue

        s = aligned["signal"]
        r = aligned["fwd_return"]

        top_mask = s >= s.quantile(top_quantile)
        bottom_mask = s <= s.quantile(bottom_quantile)

        top_returns = r[top_mask]
        bottom_returns = r[bottom_mask]
        all_returns = r

        gross_long = top_returns.mean() if len(top_returns) > 0 else 0
        gross_short = -bottom_returns.mean() if len(bottom_returns) > 0 else 0

        gross_expectancy = (gross_long + gross_short) / 2

        results[f"h{h}"] = {
            "sample_size": int(len(aligned)),
            "lead_correlation": float(s.corr(r)) if s.std() > 0 and r.std() > 0 else 0,
            "mean_forward_return": float(r.mean()),
            "median_forward_return": float(r.median()),
            "gross_expectancy": float(gross_expectancy),
            "median_gross": float((top_returns.median() - bottom_returns.median()) / 2),
            "top_quantile_return": float(top_returns.mean()),
            "bottom_quantile_return": float(bottom_returns.mean()),
            "hit_rate_long": float((top_returns > 0).mean()) if len(top_returns) > 0 else 0,
            "hit_rate_short": float((bottom_returns < 0).mean()) if len(bottom_returns) > 0 else 0,
            "payoff_ratio_long": float(top_returns[top_returns > 0].mean() / abs(top_returns[top_returns < 0].mean()))
            if len(top_returns[top_returns > 0]) > 0 and len(top_returns[top_returns < 0]) > 0
            else 0,
            "tail_loss_long": float(top_returns.quantile(0.05)),
            "tail_loss_short": float(-bottom_returns.quantile(0.95)),
            "top_count": int(top_mask.sum()),
            "bottom_count": int(bottom_mask.sum()),
        }

    return results


def run_signal_probe(
    features: pd.DataFrame,
    signal_column: str,
    horizons: list[int],
    window_splits: Optional[list[dict]] = None,
) -> dict:
    if signal_column not in features.columns:
        raise ValueError(f"Signal column '{signal_column}' not in features")

    prices = features["close"]
    signal = features[signal_column]

    forward_returns = compute_forward_returns(prices, horizons)

    full_metrics = compute_signal_metrics(signal, forward_returns, horizons)

    window_results = {}
    if window_splits:
        for w in window_splits:
            mask = (features.index >= w["start"]) & (features.index <= w["end"])
            if mask.sum() == 0:
                continue
            w_signal = signal[mask]
            w_fwd = forward_returns.loc[mask]
            window_results[w["name"]] = compute_signal_metrics(w_signal, w_fwd, horizons)

    return {
        "full_period": full_metrics,
        "windows": window_results,
    }


def verdict_from_metrics(
    metrics: dict,
    random_results: Optional[dict] = None,
    ge_tolerance: float = 0.0001,
    corr_tolerance: float = 0.1,
    min_pct_random_worse: float = 0.95,
) -> str:
    """Derive a PASS / FAIL / INCONCLUSIVE verdict from probe metrics.

    PASS         a horizon shows positive gross expectancy, a lead correlation
                 above the near-zero band, and (when available) beats the random
                 baseline at the required percentile.
    FAIL         every valid horizon has gross expectancy or lead correlation
                 inside the near-zero band — no usable edge.
    INCONCLUSIVE no valid horizons (insufficient data), or signal has structure
                 but does not clear the success criteria.
    """
    valid = {
        k: v for k, v in metrics.items()
        if isinstance(v, dict) and "error" not in v
    }
    if not valid:
        return "INCONCLUSIVE — insufficient data (no horizon produced enough samples)"

    rb = (random_results or {}).get("random_baseline", {})

    passing = []
    for h_key, m in valid.items():
        ge = m.get("gross_expectancy", 0)
        lc = m.get("lead_correlation", 0)
        beats_random = True
        comp = rb.get(h_key, {})
        if isinstance(comp, dict) and "error" not in comp and comp:
            beats_random = comp.get("pct_random_worse", 0) >= min_pct_random_worse
        if ge > ge_tolerance and abs(lc) > corr_tolerance and beats_random:
            passing.append(h_key)

    if passing:
        return (
            f"PASS — edge at {', '.join(sorted(passing))}: gross_expectancy>0, "
            f"|lead_corr|>{corr_tolerance}, beats random baseline"
        )

    edge_absent = all(
        abs(m.get("gross_expectancy", 0)) < ge_tolerance
        or abs(m.get("lead_correlation", 0)) < corr_tolerance
        for m in valid.values()
    )
    if edge_absent:
        return "FAIL — no edge: gross expectancy or lead correlation near zero across all horizons"

    return "INCONCLUSIVE — signal has structure but does not meet success criteria"
