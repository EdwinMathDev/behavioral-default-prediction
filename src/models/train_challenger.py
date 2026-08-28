"""
train_challenger.py
====================

Final XGBoost training — Credit Risk Engine.

Responsibility
--------------
Trains the XGBoost challenger on the fixed train_final.csv /
test_final.csv split (the same split used for the Logistic
Regression baseline), producing a single persisted artifact that
can be compared against the baseline under identical conditions —
same data, same metrics, same evaluation code.

This follows the 5-fold cross-validation done in
cross_validate_challenger.py, which confirmed XGBoost outperforms
the baseline consistently (4 of 5 folds) rather than by chance.
This script produces the artifact that would actually be promoted
to production if selected.

Pipeline position
------------------
    cross_validate_challenger.py -> [train_challenger.py] -> threshold optimization -> explainability

Input
-----
    data/features/train_final.csv
    data/features/test_final.csv

Output
------
    models/artifacts/xgb_challenger.joblib
    models/artifacts/xgb_challenger_metrics.json
    models/artifacts/figures/xgb_challenger_roc.png
    models/artifacts/figures/xgb_challenger_confusion_matrix.png
"""

import os
import json
import logging
import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.utils.metrics import evaluate_classifier, plot_roc_curve, plot_confusion_matrix
from src.models.cross_validate_challenger import XGB_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_NAME = "xgb_challenger"


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]

    logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_splits()

    # Mismos hiperparámetros usados en la cross-validation, para que
    # este resultado sea comparable con el que ya validamos por folds.
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    logging.info("XGBoost entrenado sobre el split final.")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_proba)

    plot_roc_curve(
        y_test, y_proba,
        save_path=os.path.join(FIGURES_DIR, f"{MODEL_NAME}_roc.png"),
        model_name="XGBoost Challenger",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=os.path.join(FIGURES_DIR, f"{MODEL_NAME}_confusion_matrix.png"),
        model_name="XGBoost Challenger",
    )

    joblib.dump(model, os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, f"{MODEL_NAME}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Modelo guardado en {ARTIFACTS_DIR}/{MODEL_NAME}.joblib")
    logging.info(f"Métricas guardadas en {ARTIFACTS_DIR}/{MODEL_NAME}_metrics.json")
    logging.info("Compara este archivo contra models/artifacts/logreg_baseline_metrics.json")
    return model, metrics


if __name__ == "__main__":
    run()