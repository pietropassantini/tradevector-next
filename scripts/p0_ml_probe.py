"""P0-ML Probe — walk-forward expanding window su funding features.

Regole PRD:
- No random split su serie temporali.
- Training sempre precedente al test.
- Embargo >= orizzonte target.
- Score decisivo: decile analysis OOS (non AUC/F1).
- ML batte baseline semplice: confronto con -funding_zscore P0 statistico.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tradevector.ml.dataset_builder import build_supervised_dataset
from tradevector.ml.decile_analysis import decile_analysis
from tradevector.ml.time_split import expanding_window_split
from tradevector.reporting.ml_report import generate_ml_report
from tradevector.validation.signal_probe import compute_forward_returns

FEATURE_COLS = [
    "funding_zscore",
    "funding_cumsum_3d",
    "funding_regime_persistence",
    "funding_extreme_flag",
    "atr_norm",
    "volume_zscore",
]

EXCLUDE_FROM_FEATURES = {
    "close", "volume", "returns", "log_returns",
    "atr", "funding_rate",
}


def load_features(symbol: str, timeframe: str, hypothesis_id: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / f"{hypothesis_id}_{symbol}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features non trovate: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded: {len(df)} righe, {df.index.min()} → {df.index.max()}")
    return df


def load_hypothesis(hypothesis_id: str) -> dict:
    path = PROJECT_ROOT / "research" / "hypotheses" / f"{hypothesis_id}.yml"
    with open(path) as f:
        return yaml.safe_load(f)


def fit_lgbm(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            class_weight="balanced",
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        logger.warning("LightGBM non disponibile, uso RandomForest")
        model = RandomForestClassifier(
            n_estimators=200, max_depth=4, random_state=42, class_weight="balanced"
        )
    model.fit(X_train, y_train)
    return model


def predict_score(model, X_test: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """Score long-short: P(up) - P(down). Invariante all'ordinamento classi."""
    probs = model.predict_proba(X_test)
    cls_list = list(classes)
    up_idx = cls_list.index(1) if 1 in cls_list else None
    down_idx = cls_list.index(-1) if -1 in cls_list else None
    if up_idx is not None and down_idx is not None:
        return probs[:, up_idx] - probs[:, down_idx]
    # Fallback binario
    return probs[:, -1] - probs[:, 0]


def run_expanding_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    fwd_col: str,
    min_train_size: int,
    test_size: int,
    embargo: int,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Expanding window: train su tutto il passato, test su ogni split.
    Ritorna (scores_oos, fwd_returns_oos, window_info).
    """
    splits = list(expanding_window_split(df, min_train_size, test_size, embargo))
    if not splits:
        raise ValueError(
            f"Dati insufficienti per walk-forward. "
            f"Righe={len(df)}, min_train={min_train_size}, test={test_size}, embargo={embargo}"
        )
    logger.info(f"Walk-forward: {len(splits)} finestre OOS")

    all_scores = []
    all_fwd = []
    window_info = []

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train = df.iloc[train_idx][feature_cols].dropna()
        y_train = df.iloc[train_idx].loc[X_train.index, target_col]
        valid_mask = y_train != 0  # escludi neutral dalla classificazione
        X_train_f = X_train[valid_mask]
        y_train_f = y_train[valid_mask]

        if len(X_train_f) < 50 or y_train_f.nunique() < 2:
            logger.warning(f"  Finestra {i+1}: train insufficiente ({len(X_train_f)} campioni), skip")
            continue

        X_test = df.iloc[test_idx][feature_cols].dropna()
        if len(X_test) == 0:
            continue

        model = fit_lgbm(X_train_f, y_train_f)
        scores = predict_score(model, X_test, model.classes_)
        fwd = df.iloc[test_idx].loc[X_test.index, fwd_col]

        all_scores.append(pd.Series(scores, index=X_test.index))
        all_fwd.append(fwd)

        ge_approx = np.mean(scores * fwd.values) if len(scores) > 0 else 0
        logger.info(
            f"  Finestra {i+1}: train={len(X_train_f)}, test={len(X_test)}, "
            f"train_end={df.index[train_idx[-1]].date()}, "
            f"test_start={df.index[test_idx[0]].date()} | GE≈{ge_approx:.6f}"
        )
        window_info.append({
            "window": i + 1,
            "train_size": len(X_train_f),
            "test_size": len(X_test),
            "train_end": str(df.index[train_idx[-1]].date()),
            "test_start": str(df.index[test_idx[0]].date()),
            "test_end": str(df.index[test_idx[-1]].date()),
            "ge_approx": float(ge_approx),
        })

    if not all_scores:
        raise ValueError("Nessun score OOS prodotto — dati insufficienti")

    oos_scores = pd.concat(all_scores).sort_index()
    oos_fwd = pd.concat(all_fwd).sort_index()
    # Allinea: stessi indici
    common = oos_scores.index.intersection(oos_fwd.index)
    return oos_scores.loc[common], oos_fwd.loc[common], window_info


def verdict_ml(decile_res: dict, window_info: list[dict]) -> str:
    top_ge = decile_res.get("top_decile_expectancy", 0)
    bottom_ge = decile_res.get("bottom_decile_expectancy", 0)
    monotonicity = decile_res.get("score_monotonicity", 0)
    n_windows = len(window_info)
    n_positive_ge = sum(1 for w in window_info if w["ge_approx"] > 0)

    if top_ge > 0 and bottom_ge < 0 and monotonicity >= 0.6 and n_positive_ge >= n_windows * 0.6:
        return (
            f"PASS — top decile GE={top_ge:.6f}, bottom GE={bottom_ge:.6f}, "
            f"monotonicity={monotonicity:.2f}, {n_positive_ge}/{n_windows} finestre positive"
        )
    if top_ge > 0 and n_positive_ge >= n_windows * 0.5:
        return (
            f"INCONCLUSIVE — top decile GE positiva ({top_ge:.6f}) ma "
            f"monotonicity={monotonicity:.2f} o stabilita' insufficiente "
            f"({n_positive_ge}/{n_windows} finestre)"
        )
    return (
        f"FAIL — top decile GE={top_ge:.6f}, monotonicity={monotonicity:.2f}, "
        f"{n_positive_ge}/{n_windows} finestre positive"
    )


def main():
    parser = argparse.ArgumentParser(description="P0-ML Probe — walk-forward expanding window")
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--target-type", default="directional",
                        choices=["directional", "volatility_expansion", "breakout"])
    parser.add_argument("--threshold", type=float, default=0.003,
                        help="Soglia directional target (default 0.3%%)")
    parser.add_argument("--min-train-size", type=int, default=2000,
                        help="Righe minime training (default 2000 = ~83gg 1h)")
    parser.add_argument("--test-size", type=int, default=2184,
                        help="Righe per finestra test (default 2184 = ~91gg 1h)")
    parser.add_argument("--embargo", type=int, default=24,
                        help="Barre embargo tra train e test (default 24 = 24h)")
    parser.add_argument("--model", default="lightgbm",
                        choices=["lightgbm", "random_forest", "logistic"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    hypothesis = load_hypothesis(args.hypothesis)
    logger.info(f"Hypothesis: {hypothesis['name']}")

    features = load_features(args.symbol, args.timeframe, args.hypothesis)

    # Dataset supervisionato
    fwd_col = f"fwd_return_{args.horizon}"
    df = build_supervised_dataset(
        features_df=features,
        horizon=args.horizon,
        target_type=args.target_type,
        threshold=args.threshold,
    )
    df[fwd_col] = features[fwd_col] if fwd_col in features.columns else (
        features["close"].shift(-args.horizon) / features["close"] - 1
    )
    df = df.dropna(subset=FEATURE_COLS + [fwd_col])

    logger.info(f"Dataset supervisionato: {len(df)} campioni")
    logger.info(f"Target dist: {df['target'].value_counts().to_dict()}")

    # Walk-forward OOS
    logger.info("Expanding window walk-forward...")
    oos_scores, oos_fwd, window_info = run_expanding_walk_forward(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col="target",
        fwd_col=fwd_col,
        min_train_size=args.min_train_size,
        test_size=args.test_size,
        embargo=args.embargo,
    )

    logger.info(f"OOS totali: {len(oos_scores)} campioni")

    # Decile analysis — metrica decisiva
    decile_res = decile_analysis(
        scores=oos_scores,
        forward_returns=oos_fwd,
        n_deciles=10,
    )

    logger.info("=== Decile Analysis OOS ===")
    logger.info(f"  Monotonicity:    {decile_res.get('score_monotonicity', 0):.2f}")
    logger.info(f"  Top decile GE:   {decile_res.get('top_decile_expectancy', 0):.6f}")
    logger.info(f"  Bottom decile GE:{decile_res.get('bottom_decile_expectancy', 0):.6f}")
    for d_key, d_val in decile_res.get("deciles", {}).items():
        logger.info(
            f"  {d_key:10s}: n={d_val['count']:4d} | "
            f"GE={d_val['mean_forward_return']:+.6f} | "
            f"hit={d_val['hit_rate']:.2%}"
        )

    verdict = verdict_ml(decile_res, window_info)
    logger.info(f"Verdict: {verdict}")

    # Output
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_ROOT / "reports" / "p0_ml"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / args.hypothesis / f"{args.symbol}_{args.timeframe}_{args.model}_h{args.horizon}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({
            "hypothesis": args.hypothesis,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "model": args.model,
            "horizon": args.horizon,
            "target_type": args.target_type,
            "threshold": args.threshold,
            "min_train_size": args.min_train_size,
            "test_size": args.test_size,
            "embargo": args.embargo,
            "oos_samples": len(oos_scores),
            "window_info": window_info,
            "decile_results": decile_res,
            "verdict": verdict,
        }, f, indent=2, default=str)

    md_path = generate_ml_report(
        hypothesis=hypothesis,
        ml_results={"windows": window_info},
        decile_results=decile_res,
        verdict=verdict,
        symbol=args.symbol,
        timeframe=args.timeframe,
        model_type=args.model,
        target_type=args.target_type,
        output_dir=output_dir,
    )
    logger.info(f"Report: {md_path}")
    logger.info(f"JSON:   {json_path}")

    return 0 if "PASS" in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
