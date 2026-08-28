"""
generate_oof_predictions.py
=============================

Out-of-fold predictions — Credit Risk Engine / behavioral-default-prediction.

Responsibility
--------------
optimize_threshold.py was choosing its decision threshold on
test_final.csv and then reporting final metrics on that same
test_final.csv — the same kind of leakage the rest of this project
was careful to avoid everywhere else (fit-only-on-train discipline).

This script fixes that by generating out-of-fold (OOF) predictions:
for every row in the dataset, its predicted probability comes from a
model that was trained on the other 4 folds and never saw that row.
This gives an honest, leakage-free probability for the ENTIRE
dataset, which can be used to choose a decision threshold without
ever touching test_final.csv.

Reuses load_data() and process_fold() from cross_validate_baseline.py
so the encode/log1p/impute/scale/SMOTE sequence is byte-for-byte the
same one already validated for leakage.

Pipeline position
------------------
    build_features.py -> [generate_oof_predictions.py] -> optimize_threshold.py (threshold selection)
                                                          -> train_baseline.py / test_final.csv (final, one-time evaluation)

Input
-----
    data/features/credit_card_features.csv

Output
------
    data/features/baseline_oof_predictions.csv  (columns: y_true, y_proba)
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from src.models.cross_validate_baseline import load_data, process_fold

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

        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        oof_proba[test_idx] = y_proba
        oof_assigned[test_idx] = True
        logging.info(f"Fold {fold_idx}/{n_folds} — predicciones OOF asignadas: {len(test_idx)}")

    assert oof_assigned.all(), "Algunas filas no recibieron una predicción out-of-fold — revisar los folds."

    oof_df = pd.DataFrame({"y_true": y.values, "y_proba": oof_proba})
    output_path = os.path.join(FEATURES_DIR, "baseline_oof_predictions.csv")
    oof_df.to_csv(output_path, index=False)
    logging.info(f"Predicciones out-of-fold guardadas en {output_path} ({len(oof_df)} filas).")
    return oof_df


if __name__ == "__main__":
    run()
