"""Paper trading ledger — records signals and tracks P&L in pseudo-live."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)

LEDGER_COLUMNS = [
    "signal_id",
    "timestamp",
    "symbol",
    "timeframe",
    "strategy_id",
    "side",
    "entry_price_theoretical",
    "entry_price_estimated",
    "exit_time",
    "exit_price_theoretical",
    "exit_price_estimated",
    "gross_pnl",
    "gross_pnl_pct",
    "net_pnl_estimated",
    "net_pnl_pct",
    "slippage_estimated",
    "status",
]


class PaperLedger:
    """Registro dei segnali paper.

    Il P&L viene tenuto sia in valuta quote sia in percentuale: il backtest P1
    lavora in rendimenti frazionari, quindi senza la colonna in percentuale il
    confronto paper/backtest è tra unità diverse e non significa nulla.
    """

    def __init__(self, ledger_path: Optional[Path] = None, taker_fee_bps: float = 4.0):
        self.ledger_path = ledger_path or Path("data/paper/ledger.parquet")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.taker_fee_bps = taker_fee_bps
        self.signals: list[dict] = []

    def record_signal(
        self,
        symbol: str,
        timeframe: str,
        strategy_id: str,
        side: str,
        entry_price: float,
        estimated_entry: float,
        slippage_bps: float = 2.0,
    ) -> str:
        # Risoluzione al secondo: due segnali nello stesso secondo condividevano
        # l'id e la deduplica in save() ne cancellava uno senza dirlo.
        signal_id = (
            f"{strategy_id}_{symbol}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_"
            f"{uuid4().hex[:4]}"
        )
        record = {
            "signal_id": signal_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_id": strategy_id,
            "side": side,
            "entry_price_theoretical": entry_price,
            "entry_price_estimated": estimated_entry,
            "exit_time": None,
            "exit_price_theoretical": None,
            "exit_price_estimated": None,
            "gross_pnl": None,
            "gross_pnl_pct": None,
            "net_pnl_estimated": None,
            "net_pnl_pct": None,
            "slippage_estimated": float(abs(entry_price - estimated_entry) / entry_price * 10000),
            "status": "open",
        }
        self.signals.append(record)
        return signal_id

    def close_signal(
        self,
        signal_id: str,
        exit_price: float,
        estimated_exit: float,
    ):
        for s in self.signals:
            if s["signal_id"] == signal_id and s["status"] == "open":
                s["exit_time"] = datetime.now(timezone.utc).isoformat()
                s["exit_price_theoretical"] = exit_price
                s["exit_price_estimated"] = estimated_exit
                direction = 1 if s["side"] == "long" else -1
                entry = s["entry_price_theoretical"]
                entry_est = s["entry_price_estimated"]
                if not entry or not entry_est:
                    logger.error(f"Prezzo di ingresso nullo per {signal_id}, P&L non calcolabile")
                    break

                # I prezzi stimati incorporano già lo slippage (2 bps per lato),
                # quindi dal netto restano da togliere solo le commissioni taker.
                fees = self.taker_fee_bps / 10000 * 2

                s["gross_pnl"] = direction * (exit_price - entry)
                s["gross_pnl_pct"] = direction * (exit_price / entry - 1)
                s["net_pnl_pct"] = direction * (estimated_exit / entry_est - 1) - fees
                s["net_pnl_estimated"] = s["net_pnl_pct"] * entry_est
                s["status"] = "closed"
                break

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.signals).reindex(columns=LEDGER_COLUMNS)

    def save(self):
        # self.signals è già lo stato completo (record precaricati + nuovi).
        # NON concatenare col file su disco: il chiamante ricarica il ledger
        # in memoria a inizio run, quindi un concat raddoppierebbe i record
        # ad ogni esecuzione.
        df = self.to_dataframe()
        # Guard difensiva: anche se il ledger venisse ricaricato/registrato
        # due volte, ogni signal_id resta una sola riga (tieni l'ultima = più avanzata).
        df = df.drop_duplicates(subset="signal_id", keep="last")
        df.to_parquet(self.ledger_path)
        logger.info(f"Ledger saved: {len(df)} records to {self.ledger_path}")

    def summary(self) -> dict:
        df = self.to_dataframe()
        closed = df[df["status"] == "closed"]
        if len(closed) == 0:
            return {
                "n_trades": 0, "total_gross_pnl": 0, "total_net_pnl": 0,
                "total_net_pct": 0, "net_expectancy_pct": 0, "win_rate": 0,
            }
        return {
            "n_trades": len(closed),
            "total_gross_pnl": float(closed["gross_pnl"].sum()),
            "total_net_pnl": float(closed["net_pnl_estimated"].sum()),
            # In percentuale: è l'unità in cui ragiona il backtest P1.
            "total_net_pct": float(closed["net_pnl_pct"].sum()),
            "gross_expectancy_pct": float(closed["gross_pnl_pct"].mean()),
            "net_expectancy_pct": float(closed["net_pnl_pct"].mean()),
            "win_rate": float((closed["net_pnl_pct"] > 0).mean()),
            "avg_slippage_bps": float(closed["slippage_estimated"].mean()),
        }
