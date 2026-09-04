"""
optimize_threshold_v2_active.py
==================================

Decision-threshold optimization for the ACTIVE v2 model — Home Credit
Default Risk.

Responsibility
--------------
Unlike optimize_threshold_home_credit.py (which hardcoded the Logistic
Regression baseline), this script reads WHICH model is active from
config/model_config_v2.json — the same single-source-of-truth discipline
used by v1's src/api/inference.py and dashboard/app.py. If the active
model is promoted or reverted again later, this script keeps working
without any code change, only a config update.

Threshold selection uses out-of-fold predictions (config's
oof_predictions_path) — never the test set. test_final_v2.csv is touched
exactly once, at the end, purely to report final performance.

Pipeline position
------------------
    generate_oof_predictions_v2_xgb.py -> [optimize_threshold_v2_active.py] -> explainability

Input
-----
    config/model_config_v2.json                (which model, which OOF file)
    data/features/<oof_predictions_path>         (for threshold selection)
    models/artifacts/<active_model_path>          (for final evaluation only)
    data/features/test_final_v2.csv                (for final evaluation only)

Output
------
    models/artifacts/figures/<model_name>_precision_recall_oof.png
    models/artifacts/threshold_optimization_report_v2_active.json
    config/model_config_v2.json  (updated in place with the chosen threshold)
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
CONFIG_PATH = os.path.join("config", "model_config_v2.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def load_oof_predictions(oof_path):
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f"{oof_path} not found. Run the matching generate_oof_predictions_v2*.py script first."
        )
    oof = pd.read_csv(oof_path)
    return oof["y_true"].values, oof["y_proba"].values


def find_f1_optimal_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx]), precisions, recalls, thresholds


def find_cost_optimal_threshold(y_true, y_proba, fn_cost, fp_cost):
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


def plot_precision_recall(precisions, recalls, thresholds, f1_threshold, cost_threshold, model_name, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision (OOF)")
    plt.plot(thresholds, recalls[:-1], label="Recall (OOF)")
    plt.axvline(f1_threshold, linestyle="--", color="green", label=f"F1-optimal ({f1_threshold:.2f})")
    plt.axvline(cost_threshold, linestyle="--", color="red", label=f"Cost-optimal ({cost_threshold:.2f})")
    plt.axvline(0.5, linestyle=":", color="gray", label="Default (0.50)")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title(f"Precision / Recall vs. Threshold — {model_name} (chosen with OOF, not test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Precision-recall curve (OOF) saved to {save_path}")


def evaluate_on_test_once(model_path, threshold_name, threshold_value):
    model = joblib.load(model_path)
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
    config = load_config()

    active_model_path = config["model"]["active_model_path"]
    model_name = os.path.splitext(os.path.basename(active_model_path))[0]
    oof_path = config["oof_predictions_path"]
    fn_cost = config["business_cost_assumptions"]["false_negative_cost"]
    fp_cost = config["business_cost_assumptions"]["false_positive_cost"]

    logging.info(f"Active model per config: {active_model_path}")
    logging.info(f"Loading OOF predictions from {oof_path} for threshold selection...")
    y_true_oof, y_proba_oof = load_oof_predictions(oof_path)

    f1_threshold, f1_at_best, precisions, recalls, thresholds = find_f1_optimal_threshold(
        y_true_oof, y_proba_oof
    )
    cost_threshold, cost_at_best = find_cost_optimal_threshold(y_true_oof, y_proba_oof, fn_cost, fp_cost)

    plot_precision_recall(
        precisions, recalls, thresholds, f1_threshold, cost_threshold, model_name,
        save_path=os.path.join(FIGURES_DIR, f"{model_name}_precision_recall_oof.png"),
    )

    logging.info(f"F1-optimal threshold (OOF): {f1_threshold:.3f} (F1 = {f1_at_best:.4f})")
    logging.info(f"Cost-optimal threshold (OOF): {cost_threshold:.3f} "
                 f"(total OOF cost = {cost_at_best:.1f}, FN_COST={fn_cost}, FP_COST={fp_cost})")

    logging.info("Evaluating on test_final_v2.csv (once, threshold already decided)...")
    default_metrics = evaluate_on_test_once(active_model_path, "default_0.5", 0.5)
    f1_metrics = evaluate_on_test_once(active_model_path, "f1_optimal", f1_threshold)
    cost_metrics = evaluate_on_test_once(active_model_path, "cost_optimal", cost_threshold)

    report = {
        "active_model": active_model_path,
        "threshold_selected_using": f"out-of-fold predictions ({oof_path})",
        "test_used_only_for": "final one-time evaluation, never for threshold selection",
        "default_threshold_0_5": default_metrics,
        "f1_optimal_threshold": f1_threshold,
        "f1_optimal_metrics_on_test": f1_metrics,
        "cost_optimal_threshold": cost_threshold,
        "cost_optimal_metrics_on_test": cost_metrics,
        "cost_assumptions": {"fn_cost": fn_cost, "fp_cost": fp_cost},
    }

    report_path = os.path.join(ARTIFACTS_DIR, "threshold_optimization_report_v2_active.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    # Update model_config_v2.json in place with the chosen (cost-optimal) threshold —
    # single source of truth, same discipline as v1's config after threshold optimization.
    import datetime
    config["decision_threshold"]["value"] = cost_threshold
    config["decision_threshold"]["selection_method"] = "cost_optimal"
    config["decision_threshold"]["chosen_on"] = datetime.date.today().isoformat()
    config["decision_threshold"]["notes"] = (
        f"Chosen using out-of-fold predictions ({oof_path}) — test_final_v2.csv touched only "
        f"once for final evaluation, never for threshold selection. Specific to {model_name}; "
        f"re-run this script if the active model changes."
    )
    save_config(config)
    logging.info(f"config/model_config_v2.json updated with decision_threshold.value = {cost_threshold}")

    return report


if __name__ == "__main__":
    run()
