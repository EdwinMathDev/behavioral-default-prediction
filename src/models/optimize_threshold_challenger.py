"""
optimize_threshold_challenger.py
=================================

Decision-threshold optimization for the XGBoost challenger —
Credit Risk Engine.

Responsibility
--------------
Mirrors optimize_threshold.py exactly, but for xgb_challenger.joblib
instead of the Logistic Regression baseline. A model's optimal
decision threshold is specific to its own probability distribution —
reusing the baseline's threshold (or the default 0.5) on a different
model produces a misleading comparison, which is exactly what
happened when both models were compared at 0.5 in train_challenger.py
(XGBoost showed worse recall only because its threshold was
unoptimized, not because it is a worse model).

Uses the same FN_COST / FP_COST business assumptions as the
baseline's threshold optimization, so both models are judged under
identical cost preferences.

Pipeline position
------------------
    train_challenger.py -> [optimize_threshold_challenger.py] -> baseline vs. challenger decision

Input
-----
    models/artifacts/xgb_challenger.joblib
    data/features/test_final.csv

Output
------
    models/artifacts/figures/xgb_challenger_precision_recall.png
    models/artifacts/threshold_optimization_report_xgb.json
"""

import os
import json
import logging
import joblib
import pandas as pd

from src.models.optimize_threshold import (
    find_f1_optimal_threshold,
    find_cost_optimal_threshold,
    plot_precision_recall,
    FN_COST,
    FP_COST,
)
from src.utils.metrics import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgb_challenger.joblib")


def load_model_and_test():
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return model, X_test, y_test


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model, X_test, y_test = load_model_and_test()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    f1_threshold, f1_at_best, precisions, recalls, thresholds = find_f1_optimal_threshold(y_true, y_proba)
    cost_threshold, cost_at_best = find_cost_optimal_threshold(y_true, y_proba)

    plot_precision_recall(
        precisions, recalls, thresholds, f1_threshold, cost_threshold,
        save_path=os.path.join(FIGURES_DIR, "xgb_challenger_precision_recall.png"),
    )

    logging.info(f"XGBoost — Threshold óptimo por F1:    {f1_threshold:.3f} (F1 = {f1_at_best:.4f})")
    logging.info(f"XGBoost — Threshold óptimo por costo: {cost_threshold:.3f} (costo total = {cost_at_best:.1f}, "
                 f"FN_COST={FN_COST}, FP_COST={FP_COST})")

    report = {
        "default_threshold_0_5": evaluate_classifier(y_true, (y_proba >= 0.5).astype(int), y_proba),
        "f1_optimal_threshold": f1_threshold,
        "f1_optimal_metrics": evaluate_classifier(y_true, (y_proba >= f1_threshold).astype(int), y_proba),
        "cost_optimal_threshold": cost_threshold,
        "cost_optimal_metrics": evaluate_classifier(y_true, (y_proba >= cost_threshold).astype(int), y_proba),
        "cost_assumptions": {"fn_cost": FN_COST, "fp_cost": FP_COST},
    }

    report_path = os.path.join(ARTIFACTS_DIR, "threshold_optimization_report_xgb.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return report


if __name__ == "__main__":
    run()