"""
cross_validate_challenger_v2.py
==================================

Challenger stability check — Home Credit Default Risk (v2).

Responsibility
--------------
Same purpose as cross_validate_baseline_v2.py, applied to the XGBoost
challenger: confirm the single-split numbers from train_challenger_v2.py
are stable, using the SAME StratifiedKFold(random_state=42) as the
baseline's cross-validation — this is what makes a paired comparison
(compare_baseline_vs_challenger_v2.py) valid: fold i here is the exact
same train/test split as fold i in cross_validate_baseline_v2.py.

Pipeline position
------------------
    build_features_v2.py -> [cross_validate_challenger_v2.py] -> compare_baseline_vs_challenger_v2.py

Input
-----
    data/features/home_credit_features.csv

Output
------
    models/artifacts/challenger_cross_validation_report_v2.json
"""

import os
import json
import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.models.cross_validate_baseline_v2 import load_data, process_fold
from src.utils.metrics import ks_statistic
from src.models.train_challenger_v2 import XGB_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARTIFACTS_DIR = "models/artifacts"
N_FOLDS = 5


def run(n_folds: int = N_FOLDS):
    X, y = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test, y_train, y_test = process_fold(X_train, X_test, y_train, y_test)

        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        ks = ks_statistic(y_test.values, y_proba)

        logging.info(f"Fold {fold_idx}/{n_folds} — AUC: {auc:.4f} | KS: {ks:.4f}")
        fold_results.append({"fold": fold_idx, "auc_roc": float(auc), "ks_statistic": float(ks)})

    aucs = [r["auc_roc"] for r in fold_results]
    kss = [r["ks_statistic"] for r in fold_results]

    summary = {
        "folds": fold_results,
        "auc_roc_mean": float(np.mean(aucs)),
        "auc_roc_std": float(np.std(aucs)),
        "ks_statistic_mean": float(np.mean(kss)),
        "ks_statistic_std": float(np.std(kss)),
    }

    logging.info(f"XGBoost v2 — AUC-ROC: {summary['auc_roc_mean']:.4f} +/- {summary['auc_roc_std']:.4f}")
    logging.info(f"XGBoost v2 — KS stat: {summary['ks_statistic_mean']:.4f} +/- {summary['ks_statistic_std']:.4f}")

    report_path = os.path.join(ARTIFACTS_DIR, "challenger_cross_validation_report_v2.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    return summary


if __name__ == "__main__":
    run()
