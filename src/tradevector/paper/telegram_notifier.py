"""Telegram notifier per P2 paper trading signals."""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning("Telegram disabilitato: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti")

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self.enabled:
            return False
        url = TELEGRAM_API.format(token=self.token, method="sendMessage")
        try:
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Telegram send error: {e}")
            return False

    def signal_open(self, symbol: str, side: str, price: float,
                    signal_value: float, strategy_id: str) -> None:
        emoji = "🟢" if side == "long" else "🔴"
        text = (
            f"{emoji} <b>SEGNALE P2 — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📌 Side:       <b>{side.upper()}</b>\n"
            f"💰 Prezzo:     <b>{price:,.2f}</b>\n"
            f"📊 Signal:     {signal_value:.4f}\n"
            f"🎯 Strategia:  {strategy_id}\n"
            f"⏱ Exit:       tra 8 ore"
        )
        self.send(text)

    def signal_close(self, symbol: str, side: str, entry: float,
                     exit_price: float, gross_pnl_pct: float) -> None:
        emoji = "✅" if gross_pnl_pct >= 0 else "❌"
        text = (
            f"{emoji} <b>TRADE CHIUSO — {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📌 Side:       {side.upper()}\n"
            f"📥 Entry:      {entry:,.2f}\n"
            f"📤 Exit:       {exit_price:,.2f}\n"
            f"💹 Gross PnL:  <b>{gross_pnl_pct:+.3%}</b>"
        )
        self.send(text)

    def neutral(self, symbol: str, price: float) -> None:
        text = (
            f"⏸ <b>NEUTRAL — {symbol}</b>\n"
            f"Prezzo: {price:,.2f} | Nessun segnale"
        )
        self.send(text)

    def weekly_summary(self, symbol: str, strategy_id: str, summary: dict) -> None:
        n = summary.get("n_trades", 0)
        gross = summary.get("total_gross_pnl", 0)
        net = summary.get("total_net_pnl", 0)
        wr = summary.get("win_rate", 0)
        emoji = "📈" if net >= 0 else "📉"
        text = (
            f"{emoji} <b>REPORT SETTIMANALE — {symbol}</b>\n"
            f"Strategia: {strategy_id}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🔢 Trade chiusi:  {n}\n"
            f"💹 Gross PnL:     {gross:+.4f}\n"
            f"💰 Net PnL:       {net:+.4f}\n"
            f"🎯 Win rate:      {wr:.1%}"
        )
        self.send(text)

    def error(self, msg: str) -> None:
        self.send(f"⚠️ <b>ERRORE P2</b>\n{msg}")
