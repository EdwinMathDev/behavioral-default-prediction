"""
optimize_threshold_home_credit.py
====================================

Decision-threshold optimization — Home Credit Default Risk (v2).

Responsibility
--------------
Same discipline as v1's optimize_threshold_v2.py: threshold selection must
happen on out-of-fold predictions only, never on the test set. test_final_v2.csv
is touched exactly once, at the very end, purely to report how the
already-decided threshold performs.

Cost assumptions (business_cost_assumptions) are the same 5:1 placeholder
ratio used in v1, since no v2-specific business cost figures have been
provided. Replace FN_COST / FP_COST below with real estimates when
available.

Pipeline position
------------------
    generate_oof_predictions_v2.py -> [optimize_threshold_home_credit.py] -> explainability / API

Input
-----
    data/features/baseline_oof_predictions_v2.csv   (for threshold selection)
    models/artifacts/logreg_baseline_v2.joblib        (for final evaluation only)
    data/features/test_final_v2.csv                    (for final evaluation only)

Output
------
    models/artifacts/figures/logreg_baseline_v2_precision_recall_oof.png
    models/artifacts/threshold_optimization_report_v2_home_credit.json
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

TARGET_COL = "TARGET"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logreg_baseline_v2.joblib")
OOF_PATH = os.path.join(FEATURES_DIR, "baseline_oof_predictions_v2.csv")

# Placeholder — same 5:1 ratio used in v1. Replace with real estimated
# costs when available.
FN_COST = 5.0
FP_COST = 1.0


def load_oof_predictions():
    if not os.path.exists(OOF_PATH):
        raise FileNotFoundError(
            f"{OOF_PATH} not found. Run first: python -m src.models.generate_oof_predictions_v2"
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
    plt.title("Precision / Recall vs. Threshold — chosen with OOF, not test (v2, Home Credit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Precision-recall curve (OOF) saved to {save_path}")


def evaluate_on_test_once(threshold_name: str, threshold_value: float):
    """Touches test_final_v2.csv EXACTLY ONCE per threshold, only to report
    — the threshold was already decided before this function, using OOF data."""
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final_v2.csv"))
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL].values

    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold_value).astype(int)

    metrics = evaluate_classifier(y_test, y_pred_test, y_proba_test)
    logging.info(f"[TEST — final evaluation, threshold '{threshold_name}'={threshold_value:.3f}] "
                 f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
                 f"F1={metrics['f1_score']:.4f}")
    return metrics


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logging.info("Loading out-of-fold predictions for threshold selection...")
    y_true_oof, y_proba_oof = load_oof_predictions()

    f1_threshold, f1_at_best, precisions, recalls, thresholds = find_f1_optimal_threshold(
        y_true_oof, y_proba_oof
    )
    cost_threshold, cost_at_best = find_cost_optimal_threshold(y_true_oof, y_proba_oof)

    plot_precision_recall(
        precisions, recalls, thresholds, f1_threshold, cost_threshold,
        save_path=os.path.join(FIGURES_DIR, "logreg_baseline_v2_precision_recall_oof.png"),
    )

    logging.info(f"F1-optimal threshold (chosen with OOF):    {f1_threshold:.3f} (F1 = {f1_at_best:.4f})")
    logging.info(f"Cost-optimal threshold (chosen with OOF): {cost_threshold:.3f} "
                 f"(total OOF cost = {cost_at_best:.1f}, FN_COST={FN_COST}, FP_COST={FP_COST})")

    logging.info("Evaluating on test_final_v2.csv (once, threshold already decided)...")
    default_metrics = evaluate_on_test_once("default_0.5", 0.5)
    f1_metrics = evaluate_on_test_once("f1_optimal", f1_threshold)
    cost_metrics = evaluate_on_test_once("cost_optimal", cost_threshold)

    report = {
        "threshold_selected_using": "out-of-fold predictions (baseline_oof_predictions_v2.csv)",
        "test_used_only_for": "final one-time evaluation, never for threshold selection",
        "default_threshold_0_5": default_metrics,
        "f1_optimal_threshold": f1_threshold,
        "f1_optimal_metrics_on_test": f1_metrics,
        "cost_optimal_threshold": cost_threshold,
        "cost_optimal_metrics_on_test": cost_metrics,
        "cost_assumptions": {"fn_cost": FN_COST, "fp_cost": FP_COST},
    }

    report_path = os.path.join(ARTIFACTS_DIR, "threshold_optimization_report_v2_home_credit.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    return report


if __name__ == "__main__":
    run()
