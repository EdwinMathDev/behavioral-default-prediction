"""
explain_model_v2.py
======================

Model explainability with SHAP — Home Credit Default Risk (v2).

Responsibility
--------------
Explains the ACTIVE v2 model's predictions (whichever model
config/model_config_v2.json currently points to — never hardcoded here) at
two levels:

    1. Global: which features drive the model's decisions overall
       (SHAP summary plot, mean |SHAP value| ranking).
    2. Local: why a *specific* applicant received their score
       (SHAP waterfall plot for individual cases) — a "reason code" /
       adverse-action explanation, same as v1.

Uses shap.Explainer (the unified, model-agnostic API) instead of
TreeExplainer/LinearExplainer directly, same fix already applied in v1's
explain_model.py — it auto-detects the right algorithm for whatever model
is active, so this script keeps working if the active model changes again
(as it already has once in v2: Logistic Regression -> XGBoost).

Pipeline position
------------------
    train_baseline_v2.py / train_challenger_v2.py -> optimize_threshold_v2_active.py -> [explain_model_v2.py] -> API (if built for v2)

Input
-----
    config/model_config_v2.json               (which model is active)
    data/features/train_final_v2.csv           (background sample for the explainer)
    data/features/test_final_v2.csv             (cases to explain)

Output
------
    models/artifacts/figures/<model_name>_shap_summary_plot.png
    models/artifacts/figures/<model_name>_shap_feature_importance.png
    models/artifacts/figures/<model_name>_shap_case_<index>.png
    models/artifacts/<model_name>_shap_global_importance.json
"""

import os
import json
import logging
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
CONFIG_PATH = os.path.join("config", "model_config_v2.json")

N_EXAMPLE_CASES = 3
BACKGROUND_SAMPLE_SIZE = 100


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_model_and_data():
    config = load_config()
    model_path = config["model"]["active_model_path"]
    target_col = config["target_column"]
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    model = joblib.load(model_path)
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final_v2.csv"))
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    background = pd.read_csv(
        os.path.join(FEATURES_DIR, "train_final_v2.csv")
    ).drop(columns=[target_col]).sample(n=BACKGROUND_SAMPLE_SIZE, random_state=42)

    logging.info(f"Model loaded from {model_path} (name: {model_name})")
    logging.info(f"Explaining {X_test.shape[0]} test cases, background of {BACKGROUND_SAMPLE_SIZE} train rows.")
    return model, model_name, X_test, y_test, background


def compute_shap_values(model, background, X_test):
    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_test)
    logging.info(f"SHAP values computed with {type(explainer).__name__}.")
    return explainer, shap_values


def plot_global_summary(shap_values, X_test, model_name):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{model_name}_shap_summary_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"SHAP summary plot ({model_name}) saved to {path}")


def plot_feature_importance(shap_values, X_test, model_name, top_n: int = 15):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importance.head(top_n)[::-1].plot(kind="barh")
    plt.xlabel("Mean |SHAP value| (average impact on prediction)")
    plt.title(f"Top {top_n} most influential features — {model_name}")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"{model_name}_shap_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"SHAP feature importance plot ({model_name}) saved to {path}")

    return importance


def plot_example_cases(shap_values, X_test, y_test, model_name, n_cases: int = N_EXAMPLE_CASES):
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=n_cases, replace=False)

    for i, idx in enumerate(sample_idx):
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"{model_name}_shap_case_{i}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Case {i} (test row #{idx}, actual target={y_test.iloc[idx]}) saved to {path}")


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model, model_name, X_test, y_test, background = load_model_and_data()
    explainer, shap_values = compute_shap_values(model, background, X_test)

    plot_global_summary(shap_values, X_test, model_name)
    importance = plot_feature_importance(shap_values, X_test, model_name)
    plot_example_cases(shap_values, X_test, y_test, model_name)

    logging.info(f"Top 10 most influential features ({model_name}):")
    for feat, val in importance.head(10).items():
        logging.info(f"  {feat}: {val:.4f}")

    report_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_shap_global_importance.json")
    with open(report_path, "w") as f:
        json.dump(importance.to_dict(), f, indent=2)
    logging.info(f"Full ranking saved to {report_path}")

    return importance


if __name__ == "__main__":
    run()
