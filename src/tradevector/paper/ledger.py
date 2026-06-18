"""Paper trading ledger — records signals and tracks P&L in pseudo-live."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
    "net_pnl_estimated",
    "slippage_estimated",
    "status",
]


class PaperLedger:
    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or Path("data/paper/ledger.parquet")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
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
        signal_id = f"{strategy_id}_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
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
            "net_pnl_estimated": None,
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
                s["gross_pnl"] = direction * (exit_price - s["entry_price_theoretical"])
                s["net_pnl_estimated"] = direction * (estimated_exit - s["entry_price_estimated"])
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
            return {"n_trades": 0, "total_gross_pnl": 0, "total_net_pnl": 0, "win_rate": 0}
        return {
            "n_trades": len(closed),
            "total_gross_pnl": float(closed["gross_pnl"].sum()),
            "total_net_pnl": float(closed["net_pnl_estimated"].sum()),
            "win_rate": float((closed["gross_pnl"] > 0).mean()),
            "avg_slippage_bps": float(closed["slippage_estimated"].mean()),
        }
