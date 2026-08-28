"""
optimize_threshold_v2.py
==========================

Decision-threshold optimization — Credit Risk Engine / behavioral-default-prediction.

Responsibility
--------------
Fixes a leakage issue in the original optimize_threshold.py: that
version chose the threshold AND reported final metrics on the same
test_final.csv, which optimistically biases the reported performance
toward the specific quirks of that one test split.

This version enforces a strict separation:
    1. Threshold selection happens ONLY on out-of-fold (OOF)
       predictions (see generate_oof_predictions.py) — probabilities
       the model never produced from data it was trained on.
    2. test_final.csv is touched EXACTLY ONCE, at the very end, only
       to report how the already-decided threshold performs on data
       that had no role whatsoever in choosing it.

The default 0.5 threshold is arbitrary — it ignores that, in credit
risk, a false negative (missing a real default) is typically far
more costly than a false positive (rejecting a good applicant).

Pipeline position
------------------
    generate_oof_predictions.py -> [optimize_threshold_v2.py] -> explainability / API

Input
-----
    data/features/baseline_oof_predictions.csv   (for threshold selection)
    models/artifacts/logreg_baseline.joblib       (for final evaluation only)
    data/features/test_final.csv                  (for final evaluation only)

Output
------
    models/artifacts/figures/logreg_baseline_precision_recall_oof.png
    models/artifacts/threshold_optimization_report_v2.json
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.utils.metrics import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logreg_baseline.joblib")
OOF_PATH = os.path.join(FEATURES_DIR, "baseline_oof_predictions.csv")

# ------------------------------------------------------------------
# Placeholders — replace with real estimated costs when available.
# The ratio between them matters more than the absolute numbers.
# ------------------------------------------------------------------
FN_COST = 5.0   # cost of missing a real default (false negative)
FP_COST = 1.0   # cost of wrongly rejecting a good client (false positive)


def load_oof_predictions():
    if not os.path.exists(OOF_PATH):
        raise FileNotFoundError(
            f"No se encontró {OOF_PATH}. Corre primero: "
            f"python -m src.models.generate_oof_predictions"
        )
    oof = pd.read_csv(OOF_PATH)
    return oof["y_true"].values, oof["y_proba"].values


def find_f1_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
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
    plt.plot(thresholds, precisions[:-1], label="Precision (OOF)")
    plt.plot(thresholds, recalls[:-1], label="Recall (OOF)")
    plt.axvline(f1_threshold, linestyle="--", color="green", label=f"F1-optimal ({f1_threshold:.2f})")
    plt.axvline(cost_threshold, linestyle="--", color="red", label=f"Cost-optimal ({cost_threshold:.2f})")
    plt.axvline(0.5, linestyle=":", color="gray", label="Default (0.50)")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall vs. Threshold — elegido con OOF, no con test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Precision-recall curve (OOF) saved to {save_path}")


def evaluate_on_test_once(threshold_name: str, threshold_value: float):
    """Toca test_final.csv UNA SOLA VEZ por threshold, solo para reportar
    — el threshold ya fue decidido antes de esta funcion, con datos OOF."""
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].values

    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold_value).astype(int)

    metrics = evaluate_classifier(y_test, y_pred_test, y_proba_test)
    logging.info(f"[TEST — evaluación final, threshold '{threshold_name}'={threshold_value:.3f}] "
                 f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
                 f"F1={metrics['f1_score']:.4f}")
    return metrics


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Selección de threshold: SOLO con predicciones OOF ──────
    logging.info("Cargando predicciones out-of-fold para elegir threshold...")
    y_true_oof, y_proba_oof = load_oof_predictions()

    f1_threshold, f1_at_best, precisions, recalls, thresholds = find_f1_optimal_threshold(
        y_true_oof, y_proba_oof
    )
    cost_threshold, cost_at_best = find_cost_optimal_threshold(y_true_oof, y_proba_oof)

    plot_precision_recall(
        precisions, recalls, thresholds, f1_threshold, cost_threshold,
        save_path=os.path.join(FIGURES_DIR, "logreg_baseline_precision_recall_oof.png"),
    )

    logging.info(f"Threshold óptimo por F1 (elegido con OOF):    {f1_threshold:.3f} (F1 = {f1_at_best:.4f})")
    logging.info(f"Threshold óptimo por costo (elegido con OOF): {cost_threshold:.3f} "
                 f"(costo total OOF = {cost_at_best:.1f}, FN_COST={FN_COST}, FP_COST={FP_COST})")

    # ── 2. Evaluación final: SOLO con test, UNA VEZ por threshold ─
    logging.info("Evaluando en test_final.csv (una sola vez, threshold ya decidido)...")
    default_metrics = evaluate_on_test_once("default_0.5", 0.5)
    f1_metrics = evaluate_on_test_once("f1_optimal", f1_threshold)
    cost_metrics = evaluate_on_test_once("cost_optimal", cost_threshold)

    report = {
        "threshold_selected_using": "out-of-fold predictions (baseline_oof_predictions.csv)",
        "test_used_only_for": "final one-time evaluation, never for threshold selection",
        "default_threshold_0_5": default_metrics,
        "f1_optimal_threshold": f1_threshold,
        "f1_optimal_metrics_on_test": f1_metrics,
        "cost_optimal_threshold": cost_threshold,
        "cost_optimal_metrics_on_test": cost_metrics,
        "cost_assumptions": {"fn_cost": FN_COST, "fp_cost": FP_COST},
    }

    report_path = os.path.join(ARTIFACTS_DIR, "threshold_optimization_report_v2.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return report


if __name__ == "__main__":
    run()
