"""
optimize_threshold.py
======================

Decision-threshold optimization — Credit Risk Engine.

Responsibility
--------------
The default 0.5 decision threshold used by predict() is an arbitrary
choice — it does not account for the fact that, in credit risk, a
false negative (approving an applicant who will default) is
typically far more costly than a false positive (rejecting an
applicant who would have paid). This script re-evaluates the
already-trained baseline model at every possible threshold and
reports two candidate operating points:

    1. F1-optimal threshold — maximizes the harmonic mean of
       precision and recall (a purely statistical criterion, useful
       when no explicit cost figures are available yet).
    2. Cost-optimal threshold — minimizes total expected cost given
       FN_COST and FP_COST below. These are placeholders: replace
       them with your institution's actual estimated cost per missed
       default vs. cost per wrongly rejected applicant.

This script does NOT retrain the model — it only re-scores the
existing test set at different cutoffs.

Pipeline position
------------------
    ... -> train_baseline.py -> [optimize_threshold.py] -> explainability / API

Input
-----
    models/artifacts/logreg_baseline.joblib
    data/features/test_final.csv

Output
------
    models/artifacts/figures/logreg_baseline_precision_recall.png
    models/artifacts/threshold_optimization_report.json
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

from src.utils.metrics import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logreg_baseline.joblib")

# ------------------------------------------------------------------
# Placeholders — replace with real estimated costs when available.
# The ratio between them matters more than the absolute numbers.
# Example reasoning: losing a full loan to default costs far more
# than the margin lost by rejecting a client who would have paid.
# ------------------------------------------------------------------
FN_COST = 5.0   # cost of missing a real default (false negative)
FP_COST = 1.0   # cost of wrongly rejecting a good client (false positive)


def load_model_and_test():
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return model, X_test, y_test


def find_f1_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns thresholds with len = len(precisions) - 1
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx]), precisions, recalls, thresholds


def find_cost_optimal_threshold(y_true, y_proba, fn_cost=FN_COST, fp_cost=FP_COST):
    candidate_thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_cost = 0.5, np.inf

    for t in candidate_thresholds:
        y_pred = (y_proba >= t).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        total_cost = fn * fn_cost + fp * fp_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    return float(best_threshold), float(best_cost)


def plot_precision_recall(precisions, recalls, thresholds, f1_threshold, cost_threshold, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.axvline(f1_threshold, linestyle="--", color="green", label=f"F1-optimal ({f1_threshold:.2f})")
    plt.axvline(cost_threshold, linestyle="--", color="red", label=f"Cost-optimal ({cost_threshold:.2f})")
    plt.axvline(0.5, linestyle=":", color="gray", label="Default (0.50)")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall vs. Threshold — Logistic Regression Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Precision-recall curve saved to {save_path}")


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model, X_test, y_test = load_model_and_test()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    f1_threshold, f1_at_best, precisions, recalls, thresholds = find_f1_optimal_threshold(y_true, y_proba)
    cost_threshold, cost_at_best = find_cost_optimal_threshold(y_true, y_proba)

    plot_precision_recall(
        precisions, recalls, thresholds, f1_threshold, cost_threshold,
        save_path=os.path.join(FIGURES_DIR, "logreg_baseline_precision_recall.png"),
    )

    logging.info(f"Threshold óptimo por F1:   {f1_threshold:.3f} (F1 = {f1_at_best:.4f})")
    logging.info(f"Threshold óptimo por costo: {cost_threshold:.3f} (costo total = {cost_at_best:.1f}, "
                 f"FN_COST={FN_COST}, FP_COST={FP_COST})")

    report = {
        "default_threshold_0_5": evaluate_classifier(y_true, (y_proba >= 0.5).astype(int), y_proba),
        "f1_optimal_threshold": f1_threshold,
        "f1_optimal_metrics": evaluate_classifier(y_true, (y_proba >= f1_threshold).astype(int), y_proba),
        "cost_optimal_threshold": cost_threshold,
        "cost_optimal_metrics": evaluate_classifier(y_true, (y_proba >= cost_threshold).astype(int), y_proba),
        "cost_assumptions": {"fn_cost": FN_COST, "fp_cost": FP_COST},
    }

    report_path = os.path.join(ARTIFACTS_DIR, "threshold_optimization_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return report


if __name__ == "__main__":
    run()