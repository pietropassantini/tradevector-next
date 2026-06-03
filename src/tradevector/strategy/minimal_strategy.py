"""Minimal strategy runner — simple threshold-based execution.

No grid search, no param sweep, no post-hoc tuning.
Just: take a validated score, apply simple entry/exit rules, measure net.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_minimal_strategy(
    features: pd.DataFrame,
    score_column: str,
    entry_threshold_long: float = 0.9,
    entry_threshold_short: float = 0.1,
    exit_bars: int = 4,
    cost_bps: float = 5.0,
    position_size: float = 1.0,
) -> dict:
    """Fixed-horizon long-short strategy aligned with P0 probe.

    Enters at bar t when signal meets threshold, holds for exit_bars bars,
    then exits regardless. Mirrors the P0 fixed-horizon measurement logic
    so that P1 gross expectancy is comparable to P0 gross expectancy.
    Overlapping signals are allowed — each is its own trade.
    """
    df = features.copy()
    prices = df["close"]
    score = df[score_column].dropna()

    score_quantile_high = score.quantile(entry_threshold_long)
    score_quantile_low = score.quantile(entry_threshold_short)

    cost_per_trade = cost_bps / 10000
    trades = []

    for i in range(len(df) - exit_bars):
        s = df[score_column].iloc[i]
        if pd.isna(s):
            continue

        if s >= score_quantile_high:
            direction = 1
        elif s <= score_quantile_low:
            direction = -1
        else:
            continue

        entry_price = prices.iloc[i]
        exit_price = prices.iloc[i + exit_bars]
        if entry_price <= 0:
            continue

        gross_ret = direction * (exit_price / entry_price - 1)
        net_ret = gross_ret - cost_per_trade * 2  # entry + exit

        trades.append({
            "entry_idx": i,
            "direction": direction,
            "gross_return": gross_ret,
            "net_return": net_ret,
        })

    if not trades:
        return {
            "n_trades": 0,
            "total_gross_return": 0.0,
            "total_net_return": 0.0,
            "gross_expectancy": 0.0,
            "net_expectancy": 0.0,
            "win_rate": 0.0,
            "gross_cost_ratio": 0.0,
        }

    gross_returns = np.array([t["gross_return"] for t in trades])
    net_returns = np.array([t["net_return"] for t in trades])

    total_gross = float(gross_returns.sum())
    total_net = float(net_returns.sum())
    total_costs = cost_per_trade * 2 * len(trades)
    win_rate = float((gross_returns > 0).mean())

    return {
        "n_trades": len(trades),
        "total_gross_return": total_gross,
        "total_net_return": total_net,
        "gross_expectancy": float(gross_returns.mean()),
        "net_expectancy": float(net_returns.mean()),
        "win_rate": win_rate,
        "gross_cost_ratio": float(abs(total_gross / total_costs)) if total_costs > 0 else float("inf"),
    }
