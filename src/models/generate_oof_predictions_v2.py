"""
generate_oof_predictions_v2.py
=================================

Out-of-fold predictions — Home Credit Default Risk (v2).

Responsibility
--------------
Same purpose as v1's generate_oof_predictions.py: generate a leakage-free
probability for every row in the dataset (each prediction made by a model
that never saw that row during training), so a decision threshold can be
chosen without ever touching the held-out test set.

Reuses load_data() and process_fold() from cross_validate_baseline_v2.py,
so the fill/encode/log1p/impute/scale sequence is identical to the one
already validated for leakage there.

Pipeline position
------------------
    build_features_v2.py -> [generate_oof_predictions_v2.py] -> optimize_threshold_home_credit.py (threshold selection)
                                                                -> train_baseline_v2.py / test_final_v2.csv (final, one-time evaluation)

Input
-----
    data/features/home_credit_features.csv

Output
------
    data/features/baseline_oof_predictions_v2.csv  (columns: y_true, y_proba)
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.models.cross_validate_baseline_v2 import load_data, process_fold

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

        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        oof_proba[test_idx] = y_proba
        oof_assigned[test_idx] = True
        logging.info(f"Fold {fold_idx}/{n_folds} — OOF predictions assigned: {len(test_idx)}")

    assert oof_assigned.all(), "Some rows did not receive an out-of-fold prediction — check the folds."

    oof_df = pd.DataFrame({"y_true": y.values, "y_proba": oof_proba})
    output_path = os.path.join(FEATURES_DIR, "baseline_oof_predictions_v2.csv")
    oof_df.to_csv(output_path, index=False)
    logging.info(f"Out-of-fold predictions saved to {output_path} ({len(oof_df)} rows).")
    return oof_df


if __name__ == "__main__":
    run()
