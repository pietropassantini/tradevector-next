"""P1 Minimal Strategy — backtest su segnale P0 validato."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.strategy.minimal_strategy import run_minimal_strategy
from tradevector.strategy.backtest import compute_equity_curve, backtest_metrics
from tradevector.reporting.p1_report import generate_p1_report


def load_features(symbol: str, timeframe: str, hypothesis_id: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / f"{hypothesis_id}_{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features non trovate: {path}")
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="P1 Minimal Strategy backtest")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--signal-column", default="oi_ma_ratio")
    parser.add_argument("--invert-signal", action="store_true")
    parser.add_argument("--horizon", type=int, default=8,
                        help="Exit dopo N barre (allineato all'horizon P0)")
    parser.add_argument("--entry-long", type=float, default=0.9,
                        help="Quantile soglia per posizione long")
    parser.add_argument("--entry-short", type=float, default=0.1,
                        help="Quantile soglia per posizione short")
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="Costo per trade in bps (default: 5bps)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    features = load_features(args.symbol, args.timeframe, args.hypothesis)

    if args.invert_signal:
        logger.info(f"Inversione segnale: {args.signal_column}")
        features = features.copy()
        features[args.signal_column] = -features[args.signal_column]

    complete = features.dropna(subset=[args.signal_column, "close"])
    logger.info(f"Righe complete per backtest: {len(complete)}")

    result = run_minimal_strategy(
        features=complete,
        score_column=args.signal_column,
        entry_threshold_long=args.entry_long,
        entry_threshold_short=args.entry_short,
        exit_bars=args.horizon,
        cost_bps=args.cost_bps,
    )

    strategy_returns = complete["close"].pct_change().fillna(0)
    strat_col = "strategy_returns_net"

    equity = compute_equity_curve(
        pd.Series(
            [result["total_net_return"] / max(result["n_trades"], 1)] * len(complete),
            index=complete.index,
        )
    )

    logger.info("=== P1 Risultati ===")
    logger.info(f"  N trades:          {result['n_trades']}")
    logger.info(f"  Total gross ret:   {result['total_gross_return']:+.4f}")
    logger.info(f"  Total net ret:     {result['total_net_return']:+.4f}")
    logger.info(f"  Gross expectancy:  {result['gross_expectancy']:+.6f}")
    logger.info(f"  Net expectancy:    {result['net_expectancy']:+.6f}")
    logger.info(f"  Win rate:          {result['win_rate']:.2%}")
    logger.info(f"  Gross/cost ratio:  {result['gross_cost_ratio']:.2f}x")

    net_positive = result["total_net_return"] > 0
    net_exp_positive = result["net_expectancy"] > 0
    if net_positive and net_exp_positive:
        logger.info("PASS — net return positivo, net expectancy positiva")
    else:
        logger.info("FAIL — net return o net expectancy non positivi")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "reports" / "p1"
    )

    signal_tag = f"inv_{args.signal_column}" if args.invert_signal else args.signal_column
    strategy_id = f"{args.hypothesis}_{signal_tag}_h{args.horizon}"

    json_path = output_dir / strategy_id / f"{args.symbol}_{args.timeframe}_backtest.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(
            {
                "hypothesis": args.hypothesis,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "signal_column": args.signal_column,
                "inverted": args.invert_signal,
                "horizon": args.horizon,
                "cost_bps": args.cost_bps,
                "results": result,
            },
            f, indent=2, default=str,
        )

    md_path = generate_p1_report(
        strategy_id=strategy_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        backtest_results=result,
        output_dir=output_dir,
    )
    logger.info(f"Report: {md_path}")
    logger.info(f"JSON:   {json_path}")

    return 0 if (net_positive and net_exp_positive) else 1


if __name__ == "__main__":
    sys.exit(main())
