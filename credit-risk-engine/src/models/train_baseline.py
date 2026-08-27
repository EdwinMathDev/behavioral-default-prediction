"""
train_baseline.py
==================

Baseline model — Credit Risk Engine.

Responsibility
--------------
Trains an interpretable Logistic Regression model on the prepared
train/test splits produced by train_pipeline.py, and evaluates it
with credit-risk-specific metrics (AUC-ROC, KS statistic, precision,
recall, confusion matrix).

This baseline exists to set a reproducible floor of performance.
Any more complex model developed later (XGBoost, LightGBM, etc.)
must be justified against this baseline — if it doesn't meaningfully
outperform a simple, fully explainable linear model, the added
complexity (and loss of transparency) isn't worth it, especially in
a domain where model decisions may need to be explained to
regulators or rejected applicants.

Pipeline position
------------------
    ... -> train_pipeline.py -> [train_baseline.py] -> explainability / API

Input
-----
    data/features/train_final.csv
    data/features/test_final.csv

Output
------
    models/artifacts/logreg_baseline.joblib
    models/artifacts/logreg_baseline_metrics.json
    models/artifacts/figures/logreg_baseline_roc.png
    models/artifacts/figures/logreg_baseline_confusion_matrix.png
"""

import os
import json
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import evaluate_classifier, plot_roc_curve, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_NAME = "logreg_baseline"


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> LogisticRegression:
    # max_iter alto porque con ~70 columnas tras encoding puede no converger con el default (100)
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Logistic Regression entrenado.")
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
        model_name="Logistic Regression Baseline",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=os.path.join(FIGURES_DIR, f"{MODEL_NAME}_confusion_matrix.png"),
        model_name="Logistic Regression Baseline",
    )

    # Guardar modelo y métricas
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Modelo guardado en {ARTIFACTS_DIR}/{MODEL_NAME}.joblib")
    logging.info(f"Métricas guardadas en {ARTIFACTS_DIR}/{MODEL_NAME}_metrics.json")
    return model, metrics


if __name__ == "__main__":
    run()