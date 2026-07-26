"""P2 Signal Scheduler — genera segnali paper trading ogni ora.

Flusso:
1. Scarica OI + klines aggiornati
2. Ricostruisce features
3. Calcola segnale corrente
4. Se threshold raggiunto: registra nel ledger + notifica Telegram
5. Chiude posizioni aperte da >= exit_bars ore + notifica Telegram
6. Salva ledger e stampa summary

Lanciare ogni ora via cron o manualmente.
Variabili ambiente richieste per Telegram:
  TELEGRAM_BOT_TOKEN=<token>
  TELEGRAM_CHAT_ID=<chat_id>
"""

import argparse
import json
import logging
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.paper.ledger import PaperLedger
from tradevector.paper.monitor import compare_to_backtest
from tradevector.paper.telegram_notifier import TelegramNotifier


def load_strategy(strategy_id: str) -> dict:
    path = PROJECT_ROOT / "config" / "strategies" / f"{strategy_id}.yml"
    with open(path) as f:
        return yaml.safe_load(f)


def refresh_data(symbol: str) -> bool:
    import subprocess
    logger.info(f"Aggiornamento dati {symbol}...")
    result = subprocess.run(
        [sys.executable,
         str(PROJECT_ROOT / "scripts" / "download_open_interest_binance.py"),
         "--symbol", symbol, "--period", "1h",
         "--download-klines", "--klines-interval", "1h"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        # Lo stdout del downloader va tenuto: è lì che finisce il motivo del
        # fallimento, e senza di esso il guasto resta invisibile nei log.
        logger.error(f"Download FALLITO (rc={result.returncode})")
        logger.error(f"  stdout: {result.stdout[-500:]}")
        logger.error(f"  stderr: {result.stderr[-500:]}")
        return False
    logger.info("Dati aggiornati.")
    return True


def check_data_freshness(symbol: str, max_age_hours: float = 3.0) -> None:
    """Blocca il run se l'OI è vecchio: ffillato diventa costante, e un segnale
    costante fa scattare `current >= q80` a ogni barra (long forzato).
    """
    from tradevector.data.loaders import load_klines, load_oi_data

    now = datetime.now(timezone.utc)
    for name, df in (("OI", load_oi_data(symbol, "1h")), ("klines", load_klines(symbol, "1h"))):
        age = (now - df.index.max()).total_seconds() / 3600
        if age > max_age_hours:
            raise RuntimeError(
                f"{name} {symbol} obsoleto: ultimo dato {df.index.max()} "
                f"({age:.1f}h fa, soglia {max_age_hours}h)"
            )
        logger.info(f"  {name}: ultimo dato {df.index.max()} ({age:.1f}h fa)")


def build_features(symbol: str) -> pd.DataFrame:
    from tradevector.data.loaders import load_klines, load_oi_data
    from tradevector.data.alignment import align_and_validate
    from tradevector.features.oi_features import build_oi_features

    candles = load_klines(symbol, "1h")
    oi_data = load_oi_data(symbol, "1h")
    oi_aligned = align_and_validate(
        external_df=oi_data, candles_df=candles,
        source_name=f"{symbol}_oi", method="last_known",
        # Oltre 2h di OI ffillato le feature diventano costanti: meglio NaN,
        # così il segnale sparisce invece di degenerare in long permanente.
        max_lookback=pd.Timedelta("2h"),
    )
    return build_oi_features(candles=candles, oi_data=oi_aligned)


def compute_signal(features: pd.DataFrame, quantile_window: int = 200) -> dict:
    if "oi_ma_ratio" not in features.columns:
        raise ValueError("oi_ma_ratio non trovata nelle features")

    signal = -features["oi_ma_ratio"].dropna()
    if len(signal) == 0:
        raise ValueError("Nessun valore valido di oi_ma_ratio")
    # dropna può nascondere un buco in coda: senza questo controllo il segnale
    # verrebbe calcolato su una barra vecchia spacciandola per l'ultima.
    if signal.index[-1] != features.index[-1]:
        raise ValueError(
            f"Segnale non disponibile sull'ultima barra: ultimo valido "
            f"{signal.index[-1]}, ultima barra {features.index[-1]}"
        )
    if len(signal) < quantile_window:
        logger.warning(f"Dati insufficienti: {len(signal)} < {quantile_window}, riduco finestra")
        quantile_window = max(50, len(signal) // 2)

    recent = signal.iloc[-quantile_window:]
    q80 = float(recent.quantile(0.80))
    q20 = float(recent.quantile(0.20))
    current = float(signal.iloc[-1])
    current_price = float(features["close"].iloc[-1])

    # Con q80 == q20 la condizione `current >= q80` è sempre vera e la strategia
    # degenera in long permanente. Non deve mai accadere se i dati sono freschi.
    if q80 - q20 < 1e-6:
        logger.error(
            f"Distribuzione degenere (q20={q20:.6f} q80={q80:.6f}): "
            "segnale privo di dispersione, nessun trade"
        )
        direction = "neutral"
    elif current >= q80:
        direction = "long"
    elif current <= q20:
        direction = "short"
    else:
        direction = "neutral"

    return {
        "timestamp": features.index[-1],
        "signal_value": current,
        "q80": q80,
        "q20": q20,
        "direction": direction,
        "price": current_price,
        "oi_ma_ratio": float(features["oi_ma_ratio"].dropna().iloc[-1]),
    }


def close_expired_positions(
    ledger: PaperLedger,
    features: pd.DataFrame,
    exit_bars: int,
    telegram: TelegramNotifier,
    dry_run: bool,
    slippage_bps: float = 2.0,
) -> int:
    now = datetime.now(timezone.utc)
    closed = 0
    for s in ledger.signals:
        if s["status"] != "open":
            continue
        entry_ts = pd.Timestamp(s["timestamp"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        hours_open = (now - entry_ts).total_seconds() / 3600
        if hours_open < exit_bars:
            continue

        exit_price = float(features["close"].iloc[-1])
        slip = slippage_bps / 10000
        slippage_factor = 1 - slip if s["side"] == "long" else 1 + slip
        direction = 1 if s["side"] == "long" else -1
        gross_pct = direction * (exit_price / s["entry_price_theoretical"] - 1)
        net_pct = gross_pct

        if not dry_run:
            ledger.close_signal(
                signal_id=s["signal_id"],
                exit_price=exit_price,
                estimated_exit=exit_price * slippage_factor,
            )
            # close_signal muta il record in place: il netto è già al netto di
            # slippage e commissioni, non va ricalcolato qui.
            net_pct = s.get("net_pnl_pct", gross_pct)
            telegram.signal_close(
                symbol=s["symbol"],
                side=s["side"],
                entry=s["entry_price_theoretical"],
                exit_price=exit_price,
                gross_pnl_pct=gross_pct,
                net_pnl_pct=net_pct,
            )

        logger.info(
            f"  {'[DRY] ' if dry_run else ''}Chiuso: {s['side']} | "
            f"{hours_open:.1f}h | gross={gross_pct:+.3%} net={net_pct:+.3%}"
        )
        closed += 1
    return closed


def print_summary(
    ledger: PaperLedger,
    backtest_expectancy: float,
    ledger_path: Path,
) -> dict:
    summary = ledger.summary()
    logger.info("=== P2 Summary ===")
    logger.info(f"  Trades chiusi:  {summary['n_trades']}")
    if summary["n_trades"] > 0:
        logger.info(f"  Net totale:     {summary['total_net_pct']:+.4%} "
                    f"({summary['total_net_pnl']:+.2f} quote)")
        logger.info(f"  Net expectancy: {summary['net_expectancy_pct']:+.6f}/trade")
        logger.info(f"  Win rate:       {summary['win_rate']:.2%}")

    if ledger_path.exists() and summary["n_trades"] > 0:
        comp = compare_to_backtest(ledger_path, backtest_expectancy)
        if "error" not in comp:
            logger.info(f"  Expectancy p.:  {comp['paper_expectancy']:+.6f}")
            logger.info(f"  Expectancy bt:  {comp['backtest_expectancy']:+.6f}")
            logger.info(f"  Deviazione:     {comp['deviation']:+.6f}")
            logger.info(f"  In tolleranza:  {comp['within_tolerance']}")
        else:
            logger.warning(f"  Confronto backtest non disponibile: {comp['error']}")
    return summary


def _report_state_path(strategy_id: str, symbol: str) -> Path:
    return PROJECT_ROOT / "data" / "paper" / f"{strategy_id}_{symbol}_report_state.json"


def _weekly_report_due(strategy_id: str, symbol: str) -> bool:
    """Vero al massimo una volta per settimana ISO."""
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    current = f"{year}-W{week:02d}"
    path = _report_state_path(strategy_id, symbol)
    if not path.exists():
        return True
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Stato report illeggibile ({e}), lo rigenero")
        return True
    return state.get("last_weekly_sent") != current


def _mark_weekly_report_sent(strategy_id: str, symbol: str) -> None:
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    path = _report_state_path(strategy_id, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "last_weekly_sent": f"{year}-W{week:02d}",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }),
        encoding="utf-8",
    )


def write_weekly_report(
    ledger: PaperLedger,
    strategy_id: str,
    symbol: str,
    telegram: TelegramNotifier,
) -> None:
    closed = [s for s in ledger.signals if s["status"] == "closed"]
    if len(closed) < 5:
        return
    closed_df = pd.DataFrame(closed)
    summary = ledger.summary()

    report_path = (
        PROJECT_ROOT / "reports" / "p2" / strategy_id
        / f"{symbol}_weekly_report.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# P2 Weekly Report — {strategy_id}",
        "",
        f"**Symbol:** {symbol}  **Generato:** {ts}",
        "",
        "## Performance",
        "",
        f"- Trades chiusi: {len(closed_df)}",
        f"- Net totale: {closed_df['net_pnl_pct'].sum():+.4%}",
        f"- Net expectancy: {closed_df['net_pnl_pct'].mean():+.6f} per trade",
        f"- Gross expectancy: {closed_df['gross_pnl_pct'].mean():+.6f} per trade",
        f"- Win rate: {(closed_df['net_pnl_pct'] > 0).mean():.2%}",
        f"- Slippage medio: {closed_df['slippage_estimated'].mean():.2f} bps",
        "",
        "## Trades",
        "",
        "| Signal ID | Side | Entry | Exit | Net |",
        "|-----------|------|-------|------|-----|",
    ]
    for _, row in closed_df.iterrows():
        exit_px = row.get("exit_price_theoretical")
        exit_str = f"{exit_px:.2f}" if pd.notna(exit_px) else "n/d"
        lines.append(
            f"| ...{str(row['signal_id'])[-12:]} | {row['side']} | "
            f"{row['entry_price_theoretical']:.2f} | {exit_str} | "
            f"{row['net_pnl_pct']:+.3%} |"
        )
    lines += ["", "---", "*Report P2 generato automaticamente.*"]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report: {report_path}")

    # Il riepilogo Telegram va inviato una volta a settimana: lo scheduler gira
    # ogni ora, quindi senza stato persistito il lunedì partirebbero 24 messaggi.
    if _weekly_report_due(strategy_id, symbol):
        telegram.weekly_summary(symbol, strategy_id, summary)
        _mark_weekly_report_sent(strategy_id, symbol)


def main():
    parser = argparse.ArgumentParser(description="P2 Signal Scheduler")
    parser.add_argument("--strategy", default="oi_relative_positioning_edge_v1")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--dry-run", action="store_true",
                        help="Calcola segnale senza scrivere ledger né inviare Telegram")
    parser.add_argument("--no-download", action="store_true",
                        help="Salta download (usa dati esistenti)")
    parser.add_argument("--notify-neutral", action="store_true",
                        help="Invia Telegram anche in caso neutral (verbose)")
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)
    exit_bars = strategy["exit"]["bars"]
    quantile_window = strategy["entry"]["quantile_window_bars"]

    # Net expectancy per trade del backtest di riferimento, nella stessa unità
    # del ledger (rendimento frazionario). Il numero di trade viene dal config:
    # era cablato a 188 e restava disallineato a ogni ri-validazione.
    key = "btc" if args.symbol.startswith("BTC") else "eth"
    p0 = strategy["p0_validation"]
    backtest_expectancy = p0[f"{key}_p1_net_return"] / p0[f"{key}_p1_n_trades"]

    # dry-run = non scrive ledger, MA invia Telegram per testare integrazione
    telegram = TelegramNotifier()

    ledger_path = (
        PROJECT_ROOT / "data" / "paper"
        / f"{args.strategy}_{args.symbol}_ledger.parquet"
    )
    ledger = PaperLedger(
        ledger_path=ledger_path,
        taker_fee_bps=strategy["cost_model"]["taker_fee_bps"],
    )

    # 1. Aggiorna dati — prima di toccare il ledger, così un ledger illeggibile
    #    non impedisce la raccolta dati (l'archivio OI non è ricostruibile).
    if not args.no_download:
        if not refresh_data(args.symbol):
            msg = f"Download dati fallito per {args.symbol} — run interrotto"
            logger.error(msg)
            telegram.error(msg)
            sys.exit(1)

    # 2. Freschezza: su dati vecchi il segnale degenera, meglio non operare
    if args.no_download:
        logger.warning("--no-download: controllo freschezza saltato")
    else:
        try:
            check_data_freshness(args.symbol)
        except Exception as e:
            msg = f"Dati non aggiornati: {e}"
            logger.error(msg)
            telegram.error(msg)
            sys.exit(1)

    if ledger_path.exists():
        existing = pd.read_parquet(ledger_path)
        ledger.signals = existing.to_dict("records")
        open_count = sum(1 for s in ledger.signals if s["status"] == "open")
        logger.info(f"Ledger: {len(ledger.signals)} totali, {open_count} aperti")

    # 3. Features
    try:
        features = build_features(args.symbol)
    except Exception as e:
        msg = f"Build features fallito: {e}"
        logger.error(msg)
        telegram.error(msg)
        sys.exit(1)

    # 3. Chiudi posizioni scadute
    n_closed = close_expired_positions(
        ledger, features, exit_bars, telegram, args.dry_run,
        slippage_bps=strategy["cost_model"]["exit_slippage_bps"],
    )
    if n_closed:
        logger.info(f"Chiuse {n_closed} posizioni")

    # 4. Segnale corrente
    sig = compute_signal(features, quantile_window)
    logger.info(
        f"Segnale {args.symbol}: {sig['direction'].upper()} | "
        f"val={sig['signal_value']:.4f} q20={sig['q20']:.4f} q80={sig['q80']:.4f} | "
        f"prezzo={sig['price']:.2f} oi_ma={sig['oi_ma_ratio']:.4f}"
    )

    # 5. Registra segnale
    open_positions = [s for s in ledger.signals if s["status"] == "open"]
    if sig["direction"] != "neutral" and len(open_positions) == 0:
        if not args.dry_run:
            entry_slip = strategy["cost_model"]["entry_slippage_bps"] / 10000
            slippage_factor = (
                1 + entry_slip if sig["direction"] == "long" else 1 - entry_slip
            )
            signal_id = ledger.record_signal(
                symbol=args.symbol,
                timeframe="1h",
                strategy_id=args.strategy,
                side=sig["direction"],
                entry_price=sig["price"],
                estimated_entry=sig["price"] * slippage_factor,
            )
            telegram.signal_open(
                symbol=args.symbol,
                side=sig["direction"],
                price=sig["price"],
                signal_value=sig["signal_value"],
                strategy_id=args.strategy,
            )
            logger.info(
                f"  REGISTRATO: {sig['direction'].upper()} @ {sig['price']:.2f} | "
                f"id=...{signal_id[-16:]}"
            )
        else:
            logger.info(f"  [DRY-RUN] {sig['direction'].upper()} @ {sig['price']:.2f}")
            telegram.send(
                f"🧪 <b>[DRY-RUN] {sig['direction'].upper()} — {args.symbol}</b>\n"
                f"Prezzo: {sig['price']:,.2f} | Signal: {sig['signal_value']:.4f}\n"
                f"<i>Nessun trade registrato (dry-run)</i>"
            )

    elif sig["direction"] != "neutral" and open_positions:
        logger.info(f"  {sig['direction'].upper()} ignorato: posizione già aperta")
    else:
        logger.info("  Neutral — attesa")
        if args.notify_neutral:
            telegram.neutral(args.symbol, sig["price"])

    # 6. Salva
    if not args.dry_run:
        ledger.save()

    summary = print_summary(ledger, backtest_expectancy, ledger_path)

    # 7. Report + Telegram settimanale
    write_weekly_report(ledger, args.strategy, args.symbol, telegram)

    return 0


if __name__ == "__main__":
    sys.exit(main())
