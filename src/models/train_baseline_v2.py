"""
train_baseline_v2.py
======================

Baseline model — Home Credit Default Risk (v2).

Responsibility
--------------
Trains an interpretable Logistic Regression model on the v2 train/test
splits (train_final_v2.csv / test_final_v2.csv), and evaluates it with the
same credit-risk-specific metrics used in v1 (AUC-ROC, KS statistic,
precision, recall, confusion matrix) — reusing src.utils.metrics as-is,
since nothing in that module is dataset-specific.

Same rationale as v1's train_baseline.py: this sets the reproducible floor
of performance. Any more complex model considered later must be justified
against this baseline with a proper statistical test (paired t-test across
shared cross-validation folds), not by which single-split number looks
higher — the same discipline that led to reverting XGBoost in v1.

Class imbalance handling
--------------------------
class_weight="balanced" is used instead of SMOTE, per the design decision
already made in train_pipeline_v2.py: v1 found SMOTE's interpolation over
one-hot encoded columns produces synthetic, meaningless fractional
category values.

Pipeline position
------------------
    ... -> train_pipeline_v2.py -> [train_baseline_v2.py] -> cross-validation / explainability

Input
-----
    data/features/train_final_v2.csv
    data/features/test_final_v2.csv

Output
------
    models/artifacts/logreg_baseline_v2.joblib
    models/artifacts/logreg_baseline_v2_metrics.json
    models/artifacts/figures/logreg_baseline_v2_roc.png
    models/artifacts/figures/logreg_baseline_v2_confusion_matrix.png
"""

import os
import json
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import evaluate_classifier, plot_roc_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "TARGET"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_NAME = "logreg_baseline_v2"


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final_v2.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final_v2.csv"))

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    logging.info("Logistic Regression (v2) trained.")
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
        model_name="Logistic Regression Baseline v2 (Home Credit)",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=os.path.join(FIGURES_DIR, f"{MODEL_NAME}_confusion_matrix.png"),
        model_name="Logistic Regression Baseline v2 (Home Credit)",
    )

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Model saved to {ARTIFACTS_DIR}/{MODEL_NAME}.joblib")
    logging.info(f"Metrics saved to {ARTIFACTS_DIR}/{MODEL_NAME}_metrics.json")
    return model, metrics


if __name__ == "__main__":
    run()
