"""Validation metrics shared across P0 and P0-ML phases."""

import numpy as np
import pandas as pd


def gross_expectancy(returns: pd.Series, signals: pd.Series) -> float:
    aligned = pd.DataFrame({"r": returns, "s": signals}).dropna()
    long_r = aligned.loc[aligned["s"] > 0, "r"]
    short_r = aligned.loc[aligned["s"] < 0, "r"]
    gross = 0
    count = 0
    if len(long_r) > 0:
        gross += long_r.mean()
        count += 1
    if len(short_r) > 0:
        gross += -short_r.mean()
        count += 1
    return float(gross / count) if count > 0 else 0.0


def hit_rate(returns: pd.Series, signals: pd.Series) -> float:
    aligned = pd.DataFrame({"r": returns, "s": signals}).dropna()
    if len(aligned) == 0:
        return 0.0
    correct = ((aligned["s"] > 0) & (aligned["r"] > 0)) | ((aligned["s"] < 0) & (aligned["r"] < 0))
    return float(correct.mean())


def payoff_ratio(returns: pd.Series, signals: pd.Series) -> float:
    aligned = pd.DataFrame({"r": returns, "s": signals}).dropna()
    correct = ((aligned["s"] > 0) & (aligned["r"] > 0)) | ((aligned["s"] < 0) & (aligned["r"] < 0))
    winners = aligned.loc[correct, "r"].abs()
    losers = aligned.loc[~correct, "r"].abs()
    if len(winners) == 0 or len(losers) == 0:
        return 0.0
    return float(winners.mean() / losers.mean())


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def profit_factor(returns: pd.Series, signals: pd.Series) -> float:
    aligned = pd.DataFrame({"r": returns, "s": signals}).dropna()
    gross = aligned["r"] * aligned["s"].apply(np.sign)
    winners = gross[gross > 0].sum()
    losers = abs(gross[gross < 0].sum())
    return float(winners / losers) if losers > 0 else float("inf")


def window_stability(
    window_metrics: dict,
    metric_key: str = "gross_expectancy",
) -> dict:
    values = []
    for _, h_metrics in window_metrics.items():
        for h_key, m in h_metrics.items():
            if isinstance(m, dict) and metric_key in m:
                values.append(m[metric_key])
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "values": []}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "values": [float(v) for v in values],
    }
