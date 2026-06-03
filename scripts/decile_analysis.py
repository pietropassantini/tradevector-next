"""Decile analysis — distribuzione del forward return per decile di segnale."""

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

from tradevector.ml.decile_analysis import decile_analysis
from tradevector.validation.signal_probe import compute_forward_returns


def load_features(symbol: str, timeframe: str, hypothesis_id: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / f"{hypothesis_id}_{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features non trovate: {path}")
    return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="Decile analysis su segnale P0")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--signal-column", default="oi_ma_ratio")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--invert-signal", action="store_true")
    parser.add_argument("--n-deciles", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    features = load_features(args.symbol, args.timeframe, args.hypothesis)

    if args.invert_signal:
        logger.info(f"Inversione segnale: {args.signal_column}")
        features = features.copy()
        features[args.signal_column] = -features[args.signal_column]

    if args.signal_column not in features.columns:
        raise ValueError(f"Colonna '{args.signal_column}' non trovata")

    fwd = compute_forward_returns(features["close"], [args.horizon])
    fwd_col = f"fwd_return_{args.horizon}"

    result = decile_analysis(
        scores=features[args.signal_column],
        forward_returns=fwd[fwd_col],
        n_deciles=args.n_deciles,
    )

    logger.info(f"Decile analysis h{args.horizon} — {args.symbol}")
    logger.info(f"  Monotonicity score: {result.get('score_monotonicity', 0):.2f}")
    logger.info(f"  Top decile GE:      {result.get('top_decile_expectancy', 0):.6f}")
    logger.info(f"  Bottom decile GE:   {result.get('bottom_decile_expectancy', 0):.6f}")

    if "deciles" in result:
        logger.info("  Decile | N  | Mean Fwd Ret | Hit Rate")
        for d_key, d_val in result["deciles"].items():
            logger.info(
                f"  {d_key:8s} | {d_val['count']:3d} | "
                f"{d_val['mean_forward_return']:+.6f}  | {d_val['hit_rate']:.2%}"
            )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "reports" / "p0" / args.hypothesis
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_tag = f"inv_{args.signal_column}" if args.invert_signal else args.signal_column
    json_path = output_dir / f"{args.symbol}_{args.timeframe}_decile_{signal_tag}_h{args.horizon}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    md_lines = [
        f"# Decile Analysis — {args.hypothesis}",
        "",
        f"**Symbol:** {args.symbol}  **Timeframe:** {args.timeframe}",
        f"**Signal:** {'−' if args.invert_signal else ''}{args.signal_column}  "
        f"**Horizon:** h{args.horizon}",
        "",
        f"- Monotonicity score: {result.get('score_monotonicity', 0):.2f} "
        f"(1.0 = perfettamente monotono)",
        f"- Top decile GE: {result.get('top_decile_expectancy', 0):.6f}",
        f"- Bottom decile GE: {result.get('bottom_decile_expectancy', 0):.6f}",
        f"- Total samples: {result.get('total_samples', 0)}",
        "",
        "| Decile | N | Mean Fwd Ret | Median | Hit Rate | Tail 5% | Tail 95% |",
        "|--------|---|-------------|--------|----------|---------|----------|",
    ]
    for d_key, d_val in result.get("deciles", {}).items():
        md_lines.append(
            f"| {d_key} | {d_val['count']} | {d_val['mean_forward_return']:+.6f} | "
            f"{d_val['median_forward_return']:+.6f} | {d_val['hit_rate']:.2%} | "
            f"{d_val['tail_5pct']:+.6f} | {d_val['tail_95pct']:+.6f} |"
        )
    md_lines += ["", "---", "*Decile analysis completata.*"]

    md_path = output_dir / f"{args.symbol}_{args.timeframe}_decile_{signal_tag}_h{args.horizon}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"Report: {md_path}")

    monotonicity = result.get("score_monotonicity", 0)
    return 0 if monotonicity >= 0.6 else 1


if __name__ == "__main__":
    sys.exit(main())
