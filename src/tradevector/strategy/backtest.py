"""Backtest engine for P1 minimal strategies."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_equity_curve(strategy_returns: pd.Series, initial_capital: float = 10000) -> pd.Series:
    return initial_capital * (1 + strategy_returns).cumprod()


def backtest_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    periods_per_year: int = 365 * 24,
) -> dict:
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = float(drawdown.min())
    sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() > 0 else 0
    return {
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "volatility": float(returns.std()),
    }
