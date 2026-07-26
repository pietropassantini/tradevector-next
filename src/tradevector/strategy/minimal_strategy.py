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
    quantile_window: Optional[int] = 200,
    max_concurrent: Optional[int] = 1,
) -> dict:
    """Strategia long-short a orizzonte fisso, allineata alla sonda P0.

    Entra alla barra t se il segnale supera la soglia, tiene per exit_bars
    barre, poi esce comunque.

    `quantile_window` è la finestra su cui si calcolano le soglie, ed è la
    stessa che usa lo scheduler in produzione: a ogni barra vede solo le
    ultime N osservazioni, quella corrente inclusa. Passare None ricalcola le
    soglie sull'intero campione — comodo per riprodurre risultati storici, ma
    è look-ahead: alla barra t quei quantili non erano ancora osservabili.

    `max_concurrent` limita le posizioni contemporanee e riflette
    risk.max_concurrent_positions. Con None i segnali si sovrappongono e ogni
    barra qualificata apre un trade: i rendimenti sommati non sono allora una
    curva di capitale percorribile, perché richiederebbero fino a exit_bars
    posizioni aperte insieme.
    """
    df = features.copy()
    prices = df["close"]
    score = df[score_column]

    soglie_in_sample = quantile_window is None
    if soglie_in_sample:
        valid = score.dropna()
        hi_full = valid.quantile(entry_threshold_long)
        lo_full = valid.quantile(entry_threshold_short)

    cost_per_trade = cost_bps / 10000
    trades = []
    aperte: list[int] = []  # indici di uscita delle posizioni ancora in corso

    for i in range(len(df) - exit_bars):
        s = score.iloc[i]
        if pd.isna(s):
            continue

        aperte = [uscita for uscita in aperte if uscita > i]
        if max_concurrent is not None and len(aperte) >= max_concurrent:
            continue

        if soglie_in_sample:
            hi, lo = hi_full, lo_full
        else:
            finestra = score.iloc[max(0, i - quantile_window + 1):i + 1].dropna()
            if len(finestra) < max(30, quantile_window // 4):
                continue
            hi = finestra.quantile(entry_threshold_long)
            lo = finestra.quantile(entry_threshold_short)
            # Stessa guardia dello scheduler: senza dispersione la condizione
            # `s >= hi` sarebbe sempre vera e la strategia degenererebbe.
            if hi - lo < 1e-9:
                continue

        if s >= hi:
            direction = 1
        elif s <= lo:
            direction = -1
        else:
            continue

        entry_price = prices.iloc[i]
        exit_price = prices.iloc[i + exit_bars]
        if entry_price <= 0:
            continue

        gross_ret = direction * (exit_price / entry_price - 1)
        net_ret = gross_ret - cost_per_trade * 2  # entry + exit

        aperte.append(i + exit_bars)
        trades.append({
            "entry_idx": i,
            "exit_idx": i + exit_bars,
            "entry_time": df.index[i],
            "exit_time": df.index[i + exit_bars],
            "direction": direction,
            "gross_return": gross_ret,
            "net_return": net_ret,
        })

    if not trades:
        return {
            "n_trades": 0,
            "total_gross_return": 0.0,
            "total_net_return": 0.0,
            "net_return_compounded": 0.0,
            "gross_expectancy": 0.0,
            "net_expectancy": 0.0,
            "win_rate": 0.0,
            "gross_cost_ratio": 0.0,
            "thresholds": "in_sample" if soglie_in_sample else f"rolling_{quantile_window}",
            "max_concurrent": max_concurrent,
        }

    gross_returns = np.array([t["gross_return"] for t in trades])
    net_returns = np.array([t["net_return"] for t in trades])

    total_gross = float(gross_returns.sum())
    total_net = float(net_returns.sum())
    total_costs = cost_per_trade * 2 * len(trades)
    win_rate = float((net_returns > 0).mean())

    # Il composto ha senso solo se i trade sono in sequenza: con posizioni
    # sovrapposte non esiste un unico capitale che li attraversa tutti.
    sequenziali = max_concurrent == 1
    net_compounded = float(np.prod(1 + net_returns) - 1) if sequenziali else float("nan")

    return {
        "n_trades": len(trades),
        "total_gross_return": total_gross,
        "total_net_return": total_net,
        "net_return_compounded": net_compounded,
        "gross_expectancy": float(gross_returns.mean()),
        "net_expectancy": float(net_returns.mean()),
        "win_rate": win_rate,
        "gross_cost_ratio": float(abs(total_gross / total_costs)) if total_costs > 0 else float("inf"),
        "thresholds": "in_sample" if soglie_in_sample else f"rolling_{quantile_window}",
        "max_concurrent": max_concurrent,
        "trades": trades,
    }
