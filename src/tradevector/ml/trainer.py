"""ML trainer module — fits models with temporal validation."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def train_evaluate_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_size: int,
    test_size: int,
    embargo: int = 0,
    model_type: str = "lightgbm",
) -> dict:
    from .time_split import walk_forward_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    splits = list(walk_forward_split(df, train_size, test_size, embargo))
    results = {"windows": [], "oos_predictions": [], "feature_importance": {}}

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_col]

        if model_type == "logistic":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, using RandomForest")
                model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        if y_prob.shape[1] >= 2:
            scores = y_prob[:, 1]
        else:
            scores = y_prob[:, 0]

        results["windows"].append({
            "window": i,
            "train_start": str(df.index[train_idx[0]]),
            "train_end": str(df.index[train_idx[-1]]),
            "test_start": str(df.index[test_idx[0]]),
            "test_end": str(df.index[test_idx[-1]]),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })
        results["oos_predictions"].extend(
            [{"timestamp": str(df.index[j]), "score": float(scores[k]),
              "target": int(y_test.iloc[k])}
             for k, j in enumerate(test_idx)]
        )

    return results
