"""
cross_validate_challenger.py
=============================

Challenger model evaluation — Credit Risk Engine.

Responsibility
--------------
Evaluates XGBoost as a challenger to the Logistic Regression
baseline, using the exact same 5-fold stratified cross-validation
methodology (same folds, same per-fold encoding/scaling/SMOTE fit
discipline) implemented in cross_validate_baseline.py, so the
comparison is apples-to-apples rather than two models scored under
different conditions.

Decision rule
--------------
The challenger only replaces the baseline as the active model if it
beats it on AUC-ROC AND KS by a margin clearly larger than the
baseline's own fold-to-fold variance (std ≈ 0.006 on AUC). A gain
smaller than that is statistical noise, not a real improvement, and
does not justify losing the baseline's direct interpretability.

Pipeline position
------------------
    cross_validate_baseline.py -> [cross_validate_challenger.py] -> decision: promote or keep baseline

Input
-----
    data/features/credit_card_features.csv

Output
------
    models/artifacts/challenger_cross_validation_report.json
"""

import os
import json
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.models.cross_validate_baseline import load_data, process_fold
from src.utils.metrics import ks_statistic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARTIFACTS_DIR = "models/artifacts"
N_FOLDS = 5

# Hiperparámetros conservadores de partida — no se optimizan todavía.
# El objetivo de este script es comparar arquitecturas (lineal vs.
# árboles), no exprimir el mejor XGBoost posible; eso viene después,
# solo si XGBoost gana esta primera comparación.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42,
)


def run(n_folds: int = N_FOLDS):
    X, y = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test, y_train, y_test = process_fold(X_train, X_test, y_train, y_test)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        ks = ks_statistic(y_test.values, y_proba)

        logging.info(f"Fold {fold_idx}/{n_folds} — AUC: {auc:.4f} | KS: {ks:.4f}")
        fold_results.append({"fold": fold_idx, "auc_roc": float(auc), "ks_statistic": float(ks)})

    aucs = [r["auc_roc"] for r in fold_results]
    kss = [r["ks_statistic"] for r in fold_results]

    summary = {
        "model": "XGBoost",
        "params": XGB_PARAMS,
        "folds": fold_results,
        "auc_roc_mean": float(np.mean(aucs)),
        "auc_roc_std": float(np.std(aucs)),
        "ks_statistic_mean": float(np.mean(kss)),
        "ks_statistic_std": float(np.std(kss)),
    }

    logging.info(f"XGBoost — AUC-ROC: {summary['auc_roc_mean']:.4f} +/- {summary['auc_roc_std']:.4f}")
    logging.info(f"XGBoost — KS stat:  {summary['ks_statistic_mean']:.4f} +/- {summary['ks_statistic_std']:.4f}")
    logging.info("Compara estos números contra models/artifacts/baseline_cross_validation_report.json")

    report_path = os.path.join(ARTIFACTS_DIR, "challenger_cross_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return summary


if __name__ == "__main__":
    run()