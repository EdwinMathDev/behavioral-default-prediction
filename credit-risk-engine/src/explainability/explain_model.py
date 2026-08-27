"""
explain_model.py
=================

Model explainability with SHAP — Credit Risk Engine.

Responsibility
--------------
Explains the active model's predictions (currently xgb_challenger,
read from config/model_config.json — never hardcoded here) at two
levels:

    1. Global: which features drive the model's decisions overall
       (SHAP summary plot, mean |SHAP value| ranking).
    2. Local: why a *specific* applicant received their score
       (SHAP waterfall plot for individual cases), which is what a
       "reason code" / adverse-action explanation requires in
       consumer lending.

This is not optional polish. A tree ensemble like XGBoost is not
self-explanatory the way Logistic Regression coefficients are —
SHAP is what restores the ability to answer "why was this applicant
scored this way", both for internal model validation and for
regulatory/consumer-facing explanations.

Pipeline position
------------------
    train_challenger.py -> optimize_threshold_challenger.py -> [explain_model.py] -> API

Input
-----
    models/artifacts/xgb_challenger.joblib  (path read from config)
    data/features/test_final.csv

Output
------
    models/artifacts/figures/shap_summary_plot.png
    models/artifacts/figures/shap_feature_importance.png
    models/artifacts/figures/shap_case_<index>.png   (a few example cases)
    models/artifacts/shap_global_importance.json
"""

import os
import json
import logging
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")

# Cuántos casos individuales explicar como ejemplo (uno de cada tipo,
# para ilustrar distintos patrones de decisión)
N_EXAMPLE_CASES = 3


def load_model_and_data():
    config = load_config()
    model_path = config["model"]["active_model_path"]
    target_col = config["target_column"]

    model = joblib.load(model_path)
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    logging.info(f"Modelo cargado desde {model_path}")
    logging.info(f"Explicando sobre {X_test.shape[0]} casos de test.")
    return model, X_test, y_test


def compute_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    logging.info("SHAP values calculados.")
    return explainer, shap_values


def plot_global_summary(shap_values, X_test):
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"SHAP summary plot guardado en {path}")


def plot_feature_importance(shap_values, X_test, top_n: int = 15):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 6))
    importance.head(top_n)[::-1].plot(kind="barh")
    plt.xlabel("Mean |SHAP value| (impacto promedio en la predicción)")
    plt.title(f"Top {top_n} variables más influyentes — XGBoost")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "shap_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"SHAP feature importance plot guardado en {path}")

    return importance


def plot_example_cases(explainer, shap_values, X_test, y_test, n_cases: int = N_EXAMPLE_CASES):
    """Explica casos individuales — el equivalente a un 'reason code'
    para un solicitante específico."""
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=n_cases, replace=False)

    for i, idx in enumerate(sample_idx):
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"shap_case_{i}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Caso {i} (fila test #{idx}, target real={y_test.iloc[idx]}) "
                     f"guardado en {path}")


def run():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    model, X_test, y_test = load_model_and_data()
    explainer, shap_values = compute_shap_values(model, X_test)

    plot_global_summary(shap_values, X_test)
    importance = plot_feature_importance(shap_values, X_test)
    plot_example_cases(explainer, shap_values, X_test, y_test)

    logging.info("Top 10 variables más influyentes:")
    for feat, val in importance.head(10).items():
        logging.info(f"  {feat}: {val:.4f}")

    report_path = os.path.join(ARTIFACTS_DIR, "shap_global_importance.json")
    with open(report_path, "w") as f:
        json.dump(importance.to_dict(), f, indent=2)
    logging.info(f"Ranking completo guardado en {report_path}")

    return importance


if __name__ == "__main__":
    run()