"""
train_challenger_v2.py
=========================

Challenger model — Home Credit Default Risk (v2).

Responsibility
--------------
Trains an XGBoost model on the same v2 train/test splits used for the
Logistic Regression baseline, to test whether a more complex, non-linear
model can close (or widen) the gap between v2 and v1's performance.

This does NOT assume XGBoost will win just because it did (narrowly, and
not significantly) show a nominal edge in v1's early informal comparison.
v1's own rigorous finding was that XGBoost did NOT significantly
outperform Logistic Regression there — this script exists to test the same
question independently on v2's genuinely different, richer dataset, not to
confirm a prior assumption.

Class imbalance handling
--------------------------
scale_pos_weight (XGBoost's native imbalance handling) is used instead of
SMOTE, consistent with the same reasoning already applied throughout v2's
pipeline (v1 found SMOTE's interpolation over one-hot columns produces
meaningless synthetic categories).

Pipeline position
------------------
    train_pipeline_v2.py -> [train_challenger_v2.py] -> cross_validate_challenger_v2.py -> compare_baseline_vs_challenger_v2.py

Input
-----
    data/features/train_final_v2.csv
    data/features/test_final_v2.csv

Output
------
    models/artifacts/xgb_challenger_v2.joblib
    models/artifacts/xgb_challenger_v2_metrics.json
    models/artifacts/figures/xgb_challenger_v2_roc.png
    models/artifacts/figures/xgb_challenger_v2_confusion_matrix.png
"""

import os
import json
import logging
import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.utils.metrics import evaluate_classifier, plot_roc_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "TARGET"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_NAME = "xgb_challenger_v2"

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
}


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final_v2.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final_v2.csv"))

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> XGBClassifier:
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    logging.info(f"Class imbalance: {n_neg} negatives, {n_pos} positives -> scale_pos_weight={scale_pos_weight:.2f}")

    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    logging.info("XGBoost (v2 challenger) trained.")
    return model


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_splits()
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_proba)

    plot_roc_curve(
        y_test, y_proba,
        save_path=os.path.join(FIGURES_DIR, f"{MODEL_NAME}_roc.png"),
        model_name="XGBoost Challenger v2 (Home Credit)",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=os.path.join(FIGURES_DIR, f"{MODEL_NAME}_confusion_matrix.png"),
        model_name="XGBoost Challenger v2 (Home Credit)",
    )

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Model saved to {ARTIFACTS_DIR}/{MODEL_NAME}.joblib")
    logging.info(f"Metrics saved to {ARTIFACTS_DIR}/{MODEL_NAME}_metrics.json")
    return model, metrics


if __name__ == "__main__":
    run()
