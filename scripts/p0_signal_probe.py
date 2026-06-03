"""P0 Statistical Signal Probe — main entry point."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.validation.signal_probe import run_signal_probe, verdict_from_metrics
from tradevector.validation.random_baseline import run_random_baseline
from tradevector.reporting.p0_report import generate_p0_report


def load_hypothesis(hypothesis_id: str) -> dict:
    hyp_path = PROJECT_ROOT / "research" / "hypotheses" / f"{hypothesis_id}.yml"
    if not hyp_path.exists():
        raise FileNotFoundError(f"Hypothesis not found: {hyp_path}")
    with open(hyp_path) as f:
        return yaml.safe_load(f)


def load_feature_data(symbol: str, timeframe: str, hypothesis_id: str) -> pd.DataFrame:
    feature_path = (
        PROJECT_ROOT / "data" / "features"
        / f"{hypothesis_id}_{symbol}_{timeframe}.parquet"
    )
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Features not found: {feature_path}\n"
            f"Run feature builder first: python scripts/build_oi_features.py"
        )
    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded features: {len(df)} rows, {len(df.columns)} cols")
    return df


def main():
    parser = argparse.ArgumentParser(description="P0 Statistical Signal Probe")
    parser.add_argument("--hypothesis", required=True, help="Hypothesis ID")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Data timeframe")
    parser.add_argument("--leads", default="1,2,4,8", help="Comma-separated lead horizons")
    parser.add_argument("--signal-column", default="oi_zscore", help="Column to use as signal")
    parser.add_argument("--windows", default=None, help="Comma-separated window names or 'all'")
    parser.add_argument("--n-random-trials", type=int, default=100, help="Random baseline trials")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--invert-signal", action="store_true",
                        help="Negate the signal before probing (test anti-correlated direction)")

    args = parser.parse_args()

    hypothesis = load_hypothesis(args.hypothesis)
    logger.info(f"Hypothesis: {hypothesis['name']} ({args.hypothesis})")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}")

    leads = [int(x.strip()) for x in args.leads.split(",")]
    logger.info(f"Leads: {leads}")

    features = load_feature_data(args.symbol, args.timeframe, args.hypothesis)

    if args.invert_signal:
        logger.info(f"Inverting signal: {args.signal_column} → -{args.signal_column}")
        features = features.copy()
        features[args.signal_column] = -features[args.signal_column]

    window_splits = None
    if args.windows:
        experiments_path = PROJECT_ROOT / "config" / "experiments.yml"
        if experiments_path.exists():
            with open(experiments_path) as f:
                experiments = yaml.safe_load(f)
            exp_key = f"{args.hypothesis}_{args.symbol.lower()}_{args.timeframe}"
            exp_config = experiments.get("experiments", {}).get(exp_key, {})
            exp_windows = exp_config.get("windows", [])
            if exp_windows:
                window_splits = [
                    {"name": w["name"], "start": w["test_start"], "end": w["test_end"]}
                    for w in exp_windows
                ]

    logger.info("Running signal probe...")
    probe_results = run_signal_probe(
        features=features,
        signal_column=args.signal_column,
        horizons=leads,
        window_splits=window_splits,
    )

    full_metrics = probe_results["full_period"]
    for h_key, h_metrics in full_metrics.items():
        if isinstance(h_metrics, dict) and "error" not in h_metrics:
            logger.info(
                f"  {h_key}: gross_expectancy={h_metrics['gross_expectancy']:.6f}, "
                f"lead_corr={h_metrics['lead_correlation']:.6f}, "
                f"n={h_metrics['sample_size']}"
            )
        else:
            logger.warning(f"  {h_key}: {h_metrics}")

    logger.info("Running random baseline comparison...")
    random_results = run_random_baseline(
        features=features,
        signal_column=args.signal_column,
        horizons=leads,
        n_trials=args.n_random_trials,
    )

    for h_key, comp in random_results["random_baseline"].items():
        if "error" not in comp:
            logger.info(
                f"  {h_key}: actual_ge={comp['actual_gross_expectancy']:.6f}, "
                f"random_mean_ge={comp['random_mean_gross_expectancy']:.6f}, "
                f"pct_better={comp['pct_random_worse']:.1%}"
            )

    window_results = probe_results.get("windows", {})
    if window_results:
        logger.info("Window stability:")
        for w_name, w_metrics in window_results.items():
            for h_key, m in w_metrics.items():
                if isinstance(m, dict) and "error" not in m:
                    logger.info(
                        f"  {w_name} {h_key}: GE={m['gross_expectancy']:.6f}, "
                        f"corr={m['lead_correlation']:.4f}, n={m['sample_size']}"
                    )

    verdict = verdict_from_metrics(full_metrics, random_results=random_results)
    logger.info(f"Verdict: {verdict}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "reports" / "p0" / args.hypothesis
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{args.symbol}_{args.timeframe}_signal_probe.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "hypothesis": args.hypothesis,
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "signal_column": args.signal_column,
                "leads": leads,
                "probe": probe_results,
                "random_baseline": random_results,
                "verdict": verdict,
            },
            f,
            indent=2,
            default=str,
        )

    md_path = generate_p0_report(
        hypothesis=hypothesis,
        probe_results=probe_results,
        random_results=random_results,
        verdict=verdict,
        symbol=args.symbol,
        timeframe=args.timeframe,
        signal_column=args.signal_column,
        output_dir=output_dir,
    )
    logger.info(f"Report: {md_path}")
    logger.info(f"JSON: {json_path}")

    return 0 if "FAIL" not in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
