"""
cross_validate_baseline_v2.py
================================

Baseline stability check — Home Credit Default Risk (v2).

Responsibility
--------------
Same purpose as v1's cross_validate_baseline.py: confirm the baseline's
single-split numbers (train_baseline_v2.py) aren't a fluke of one
particular train/test division, by re-running the full fit-only-on-train
pipeline independently across 5 stratified folds.

Uses the full engineered feature set (data/features/home_credit_features.csv,
pre-split) rather than the already-split train_final_v2.csv, so each fold
gets its own independent fill/encode/transform/impute/scale fit — matching
exactly how train_pipeline_v2.py avoids leakage, repeated 5 times.

No SMOTE (class_weight='balanced' instead), consistent with the design
decision already made in train_pipeline_v2.py and train_baseline_v2.py.

Pipeline position
------------------
    build_features_v2.py -> [cross_validate_baseline_v2.py] -> train_pipeline_v2.py / train_baseline_v2.py

Input
-----
    data/features/home_credit_features.csv

Output
------
    models/artifacts/baseline_cross_validation_report_v2.json
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.metrics import ks_statistic
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "TARGET"
ID_COLS = ["SK_ID_CURR"]
CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
SKEWED_COLS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "BUREAU_TOTAL_DEBT"]

FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
N_FOLDS = 5


def load_data():
    df = pd.read_csv(os.path.join(FEATURES_DIR, "home_credit_features.csv"))
    X = df.drop(columns=[TARGET_COL] + ID_COLS)
    y = df[TARGET_COL]
    return X, y


def process_fold(X_train, X_test, y_train, y_test):
    """Replicates, for a single fold, the exact same transformation order
    as train_pipeline_v2.py: fill missing categoricals -> encode ->
    sign-aware log1p -> impute -> scale, all fit only on train."""
    X_train, X_test = X_train.copy(), X_test.copy()

    present_cat = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    X_train[present_cat] = X_train[present_cat].fillna("Missing")
    X_test[present_cat] = X_test[present_cat].fillna("Missing")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[present_cat])
    train_enc = pd.DataFrame(encoder.transform(X_train[present_cat]),
                              columns=encoder.get_feature_names_out(present_cat), index=X_train.index)
    test_enc = pd.DataFrame(encoder.transform(X_test[present_cat]),
                             columns=encoder.get_feature_names_out(present_cat), index=X_test.index)
    X_train = pd.concat([X_train.drop(columns=present_cat), train_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=present_cat), test_enc], axis=1)

    present_skewed = [c for c in SKEWED_COLS if c in X_train.columns]
    for col in present_skewed:
        X_train[col] = np.sign(X_train[col]) * np.log1p(np.abs(X_train[col]))
        X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test


def run(n_folds: int = N_FOLDS):
    X, y = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test, y_train, y_test = process_fold(X_train, X_test, y_train, y_test)

        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
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

    logging.info(f"AUC-ROC: {summary['auc_roc_mean']:.4f} +/- {summary['auc_roc_std']:.4f}")
    logging.info(f"KS stat: {summary['ks_statistic_mean']:.4f} +/- {summary['ks_statistic_std']:.4f}")

    report_path = os.path.join(ARTIFACTS_DIR, "baseline_cross_validation_report_v2.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    return summary


if __name__ == "__main__":
    run()
