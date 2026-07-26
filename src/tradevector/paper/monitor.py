"""Paper trading monitor."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compare_to_backtest(
    paper_ledger_path: Path,
    backtest_expectancy: float,
) -> dict:
    if not paper_ledger_path.exists():
        return {"error": "paper ledger not found"}

    df = pd.read_parquet(paper_ledger_path)
    closed = df[df["status"] == "closed"]
    if len(closed) == 0:
        return {"error": "no closed trades"}

    # backtest_expectancy è un rendimento frazionario per trade: va confrontato
    # con net_pnl_pct, non con gross_pnl (che è in valuta quote).
    if "net_pnl_pct" not in closed.columns or closed["net_pnl_pct"].isna().all():
        return {"error": "ledger senza net_pnl_pct: formato precedente, non confrontabile"}

    paper_expectancy = closed["net_pnl_pct"].mean()
    deviation = paper_expectancy - backtest_expectancy

    return {
        "paper_trades": len(closed),
        "paper_expectancy": float(paper_expectancy),
        "backtest_expectancy": backtest_expectancy,
        "deviation": float(deviation),
        "within_tolerance": abs(deviation / max(abs(backtest_expectancy), 0.0001)) < 0.5
        if backtest_expectancy != 0 else False,
    }
