"""P&L del ledger paper: unità e costi.

Il confronto paper/backtest è significativo solo se il ledger espone il
rendimento frazionario al netto di slippage e commissioni.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradevector.paper.ledger import PaperLedger

ENTRY_SLIP = 0.0002
EXIT_SLIP = 0.0002
FEES = 0.0008  # 4 bps taker per lato


@pytest.fixture
def ledger(tmp_path):
    return PaperLedger(ledger_path=tmp_path / "ledger.parquet", taker_fee_bps=4.0)


def _open_and_close(ledger, side, entry, exit_price):
    entry_est = entry * (1 + ENTRY_SLIP) if side == "long" else entry * (1 - ENTRY_SLIP)
    exit_est = exit_price * (1 - EXIT_SLIP) if side == "long" else exit_price * (1 + EXIT_SLIP)
    sid = ledger.record_signal(
        symbol="BTCUSDT", timeframe="1h", strategy_id="test",
        side=side, entry_price=entry, estimated_entry=entry_est,
    )
    ledger.close_signal(signal_id=sid, exit_price=exit_price, estimated_exit=exit_est)
    return ledger.signals[-1]


def test_long_gross_pct_is_fractional_return(ledger):
    rec = _open_and_close(ledger, "long", 100.0, 110.0)
    assert rec["gross_pnl_pct"] == pytest.approx(0.10)
    assert rec["gross_pnl"] == pytest.approx(10.0)


def test_short_profits_when_price_falls(ledger):
    rec = _open_and_close(ledger, "short", 100.0, 90.0)
    assert rec["gross_pnl_pct"] == pytest.approx(0.10)
    assert rec["net_pnl_pct"] > 0


def test_net_subtracts_both_slippage_and_fees(ledger):
    rec = _open_and_close(ledger, "long", 100.0, 110.0)
    # slippage già dentro i prezzi stimati, commissioni tolte a parte
    atteso = (110.0 * (1 - EXIT_SLIP)) / (100.0 * (1 + ENTRY_SLIP)) - 1 - FEES
    assert rec["net_pnl_pct"] == pytest.approx(atteso)
    assert rec["net_pnl_pct"] < rec["gross_pnl_pct"]


def test_flat_trade_is_a_loss_after_costs(ledger):
    """Entrata e uscita allo stesso prezzo devono chiudere in perdita."""
    rec = _open_and_close(ledger, "long", 100.0, 100.0)
    assert rec["gross_pnl_pct"] == pytest.approx(0.0)
    assert rec["net_pnl_pct"] < -FEES


def test_summary_expectancy_is_in_fractional_units(ledger):
    _open_and_close(ledger, "long", 100.0, 110.0)
    _open_and_close(ledger, "long", 100.0, 95.0)
    s = ledger.summary()
    assert s["n_trades"] == 2
    # ordine di grandezza di un rendimento, non di un prezzo
    assert abs(s["net_expectancy_pct"]) < 1.0
    assert s["win_rate"] == pytest.approx(0.5)


def test_segnali_ravvicinati_non_si_sovrascrivono(ledger):
    """save() deduplica per signal_id: id collidenti = record persi in silenzio."""
    import pandas as pd

    for _ in range(3):
        _open_and_close(ledger, "long", 100.0, 101.0)
    ids = [s["signal_id"] for s in ledger.signals]
    assert len(set(ids)) == 3

    ledger.save()
    assert len(pd.read_parquet(ledger.ledger_path)) == 3


def test_open_trade_has_no_pnl(ledger):
    ledger.record_signal(
        symbol="BTCUSDT", timeframe="1h", strategy_id="test",
        side="long", entry_price=100.0, estimated_entry=100.02,
    )
    rec = ledger.signals[-1]
    assert rec["status"] == "open"
    assert rec["net_pnl_pct"] is None
    assert ledger.summary()["n_trades"] == 0
