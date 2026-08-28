"""
cross_validate_baseline.py
===========================

Baseline stability check — Credit Risk Engine.

Responsibility
--------------
A single train/test split can make a model look better or worse than
it really is, purely by chance in how the data happened to be
divided. Before this baseline is used as the reference point to
justify (or reject) more complex challenger models, this script
confirms it behaves consistently across 5 independent stratified
folds.

If AUC-ROC or KS vary widely across folds, the single-split numbers
reported by train_baseline.py should not be trusted as-is, and the
cause (likely too few samples in some fold, or a feature with
unstable behavior) should be investigated before moving forward.

Note: this uses the full engineered feature set (data/features/
credit_card_features.csv, pre-split) rather than the already-split
train_final.csv, so that each fold gets its own independent
encoding/scaling/SMOTE fit — matching exactly how train_pipeline.py
avoids leakage, but repeated 5 times.

Pipeline position
------------------
    build_features.py -> [cross_validate_baseline.py] -> train_pipeline.py / train_baseline.py

Input
-----
    data/features/credit_card_features.csv

Output
------
    models/artifacts/baseline_cross_validation_report.json
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

from src.utils.metrics import ks_statistic
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
CATEGORICAL_COLS = ["EDUCATION", "MARRIAGE", "age_group"]  # SEX removida — ver FAIRNESS.md
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
N_FOLDS = 5


def load_data():
    df = pd.read_csv(os.path.join(FEATURES_DIR, "credit_card_features.csv"))
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def process_fold(X_train, X_test, y_train, y_test):
    """Replica, para un solo fold, exactamente el mismo orden de
    transformaciones que train_pipeline.py: encode -> log1p -> impute
    -> scale -> SMOTE, todo con fit solo en train."""
    X_train, X_test = X_train.copy(), X_test.copy()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[CATEGORICAL_COLS])
    train_enc = pd.DataFrame(encoder.transform(X_train[CATEGORICAL_COLS]),
                              columns=encoder.get_feature_names_out(CATEGORICAL_COLS), index=X_train.index)
    test_enc = pd.DataFrame(encoder.transform(X_test[CATEGORICAL_COLS]),
                             columns=encoder.get_feature_names_out(CATEGORICAL_COLS), index=X_test.index)
    X_train = pd.concat([X_train.drop(columns=CATEGORICAL_COLS), train_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=CATEGORICAL_COLS), test_enc], axis=1)

    pay_amt_cols = [c for c in X_train.columns if "PAY_AMT" in c]
    for col in pay_amt_cols:
        X_train[col] = np.log1p(X_train[col].clip(lower=0))
        X_test[col] = np.log1p(X_test[col].clip(lower=0))

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def run(n_folds: int = N_FOLDS):
    X, y = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test, y_train, y_test = process_fold(X_train, X_test, y_train, y_test)

        model = LogisticRegression(max_iter=2000, random_state=42)
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

    report_path = os.path.join(ARTIFACTS_DIR, "baseline_cross_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return summary


if __name__ == "__main__":
    run()
