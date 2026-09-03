"""
fairness_check_gender_v2.py
==============================

Fairness audit — Home Credit Default Risk (v2).

Responsibility
--------------
CODE_GENDER was deliberately kept in the v2 pipeline (see train_pipeline_v2.py
docstring) rather than excluded upfront — following the same evidence-first
process used for SEX in v1: audit BEFORE deciding, not the other way around.

This script answers the only question that matters: how much predictive
performance, if any, is lost by excluding CODE_GENDER from the model? If the
AUC drop is negligible, there is no performance justification for keeping a
protected attribute in a lending model.

Correction versus v1's fairness_check_sex.py
-----------------------------------------------
v1's fairness ablation was run with XGBoost, even though the eventual
production model was Logistic Regression — a mismatch that was noted but
never mattered in practice since the SEX exclusion was already baked in
upstream by the time it was caught. This script does not repeat that
mismatch: it ablates using the SAME model type actually being considered
for v2 production (Logistic Regression, class_weight='balanced'), matching
train_baseline_v2.py exactly.

This script does NOT retrain the full pipeline from scratch — it reuses
train_final_v2.csv / test_final_v2.csv (already encoded/scaled) and simply
drops the CODE_GENDER_* columns before training, so the comparison is a
clean, isolated ablation test.

Pipeline position
------------------
    train_baseline_v2.py -> [fairness_check_gender_v2.py] -> decision: retrain without CODE_GENDER or keep + document rationale

Input
-----
    data/features/train_final_v2.csv
    data/features/test_final_v2.csv

Output
------
    models/artifacts/fairness_check_gender_v2_report.json
"""

import os
import json
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import evaluate_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "TARGET"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
AUC_DELTA_THRESHOLD = 0.005  # same bar used in v1's fairness_check_sex.py


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final_v2.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final_v2.csv"))
    return train, test


def train_and_evaluate(train, test, drop_cols):
    X_train = train.drop(columns=[TARGET_COL] + drop_cols)
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL] + drop_cols)
    y_test = test[TARGET_COL]

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return evaluate_classifier(y_test, y_pred, y_proba)


def run():
    train, test = load_splits()

    gender_cols = [c for c in train.columns if c.startswith("CODE_GENDER_")]
    if not gender_cols:
        logging.warning("No CODE_GENDER_* columns found in train_final_v2.csv. "
                         "Check the actual column names with: print(train.columns.tolist())")
        return None

    logging.info(f"Found gender columns: {gender_cols}")

    logging.info("Training model WITH CODE_GENDER (reference)...")
    with_gender_metrics = train_and_evaluate(train, test, drop_cols=[])

    logging.info(f"Training model WITHOUT {gender_cols}...")
    without_gender_metrics = train_and_evaluate(train, test, drop_cols=gender_cols)

    auc_with = with_gender_metrics["auc_roc"]
    auc_without = without_gender_metrics["auc_roc"]
    auc_delta = auc_with - auc_without

    logging.info(f"AUC-ROC WITH gender:    {auc_with:.4f}")
    logging.info(f"AUC-ROC WITHOUT gender: {auc_without:.4f}")
    logging.info(f"Difference:             {auc_delta:+.4f}")

    if auc_delta < AUC_DELTA_THRESHOLD:
        logging.info(f"=> AUC loss is negligible (<{AUC_DELTA_THRESHOLD}). "
                      f"No performance justification for keeping CODE_GENDER in the model.")
    else:
        logging.info(f"=> AUC loss is notable (>={AUC_DELTA_THRESHOLD}). "
                      f"An explicit business/legal decision is required on whether the "
                      f"performance gain justifies the compliance risk.")

    report = {
        "with_gender": with_gender_metrics,
        "without_gender": without_gender_metrics,
        "auc_delta": auc_delta,
        "columns_dropped": gender_cols,
        "auc_delta_threshold_used": AUC_DELTA_THRESHOLD,
    }

    report_path = os.path.join(ARTIFACTS_DIR, "fairness_check_gender_v2_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    return report


if __name__ == "__main__":
    run()
