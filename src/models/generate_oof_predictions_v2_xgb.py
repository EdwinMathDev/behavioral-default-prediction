"""
generate_oof_predictions_v2_xgb.py
=====================================

Out-of-fold predictions for the promoted challenger — Home Credit Default
Risk (v2).

Responsibility
--------------
Same purpose as generate_oof_predictions_v2.py (Logistic Regression), but
for XGBoost — the model promoted to production per
config/model_config_v2.json's promotion_notes. Reuses load_data() and
process_fold() from cross_validate_baseline_v2.py, so the transformation
sequence is identical to the one already validated for leakage.

Pipeline position
------------------
    build_features_v2.py -> [generate_oof_predictions_v2_xgb.py] -> optimize_threshold_v2_active.py

Input
-----
    data/features/home_credit_features.csv

Output
------
    data/features/xgb_challenger_v2_oof_predictions.csv  (columns: y_true, y_proba)
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.models.cross_validate_baseline_v2 import load_data, process_fold
from src.models.train_challenger_v2 import XGB_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_DIR = "data/features"
N_FOLDS = 5


def run(n_folds: int = N_FOLDS):
    X, y = load_data()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_proba = np.zeros(len(y))
    oof_assigned = np.zeros(len(y), dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test, y_train, y_test = process_fold(X_train, X_test, y_train, y_test)

        n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=n_neg / n_pos)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        oof_proba[test_idx] = y_proba
        oof_assigned[test_idx] = True
        logging.info(f"Fold {fold_idx}/{n_folds} — OOF predictions assigned: {len(test_idx)}")

    assert oof_assigned.all(), "Some rows did not receive an out-of-fold prediction — check the folds."

    oof_df = pd.DataFrame({"y_true": y.values, "y_proba": oof_proba})
    output_path = os.path.join(FEATURES_DIR, "xgb_challenger_v2_oof_predictions.csv")
    oof_df.to_csv(output_path, index=False)
    logging.info(f"Out-of-fold predictions saved to {output_path} ({len(oof_df)} rows).")
    return oof_df


if __name__ == "__main__":
    run()
