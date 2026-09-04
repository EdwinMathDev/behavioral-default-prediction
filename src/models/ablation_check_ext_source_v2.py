"""
ablation_check_ext_source_v2.py
==================================

Feature-origin ablation — Home Credit Default Risk (v2).

Responsibility
--------------
SHAP analysis of the active v2 model (explain_model_v2.py) showed
EXT_SOURCE_1/2/3 — precomputed external scores present directly in
application_train.csv — dominating the top of the feature importance
ranking, well above any feature engineered from the 5-table relational
data in this project's own build_features_v2.py.

This script answers a question of intellectual honesty, not fairness or
compliance: how much of the model's AUC comes from these three
precomputed external scores versus from the multi-table feature
engineering actually built for this project? Uses the SAME model type
currently in production (XGBoost, config/model_config_v2.json) for the
ablation — same discipline already applied in fairness_check_gender_v2.py.

This does NOT retrain the full pipeline from scratch — it reuses
train_final_v2.csv / test_final_v2.csv (already encoded/scaled) and simply
drops the EXT_SOURCE_* columns before training.

Pipeline position
------------------
    train_challenger_v2.py -> [ablation_check_ext_source_v2.py] -> honest reporting of what the model actually learned from

Input
-----
    data/features/train_final_v2.csv
    data/features/test_final_v2.csv

Output
------
    models/artifacts/ablation_ext_source_v2_report.json
"""

import os
import json
import logging
import pandas as pd
from xgboost import XGBClassifier

from src.utils.metrics import evaluate_classifier
from src.models.train_challenger_v2 import XGB_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "TARGET"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
EXT_SOURCE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final_v2.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final_v2.csv"))
    return train, test


def train_and_evaluate(train, test, drop_cols):
    X_train = train.drop(columns=[TARGET_COL] + drop_cols)
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL] + drop_cols)
    y_test = test[TARGET_COL]

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return evaluate_classifier(y_test, y_pred, y_proba)


def run():
    train, test = load_splits()

    present_cols = [c for c in EXT_SOURCE_COLS if c in train.columns]
    if not present_cols:
        logging.warning(f"None of {EXT_SOURCE_COLS} found in train_final_v2.csv.")
        return None

    logging.info(f"Found EXT_SOURCE columns: {present_cols}")

    logging.info("Training WITH EXT_SOURCE_1/2/3 (reference — same as production model)...")
    with_ext_metrics = train_and_evaluate(train, test, drop_cols=[])

    logging.info(f"Training WITHOUT {present_cols} (only this project's own engineered features)...")
    without_ext_metrics = train_and_evaluate(train, test, drop_cols=present_cols)

    auc_with = with_ext_metrics["auc_roc"]
    auc_without = without_ext_metrics["auc_roc"]
    auc_delta = auc_with - auc_without

    logging.info(f"AUC-ROC WITH EXT_SOURCE_1/2/3:    {auc_with:.4f}")
    logging.info(f"AUC-ROC WITHOUT EXT_SOURCE_1/2/3: {auc_without:.4f}")
    logging.info(f"Difference (attributable to these 3 columns alone): {auc_delta:+.4f}")

    v1_auc = 0.7664  # v1 baseline, 5-fold CV mean, for context
    logging.info(f"For context — v1 baseline AUC (5-fold CV): {v1_auc:.4f}")
    logging.info(f"v2 WITHOUT EXT_SOURCE vs v1: {auc_without - v1_auc:+.4f}")

    report = {
        "with_ext_source": with_ext_metrics,
        "without_ext_source": without_ext_metrics,
        "auc_delta_from_ext_source_alone": auc_delta,
        "columns_tested": present_cols,
        "v1_baseline_auc_for_context": v1_auc,
    }

    report_path = os.path.join(ARTIFACTS_DIR, "ablation_ext_source_v2_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Report saved to {report_path}")

    return report


if __name__ == "__main__":
    run()
