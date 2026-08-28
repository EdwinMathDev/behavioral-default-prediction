"""
fairness_check_sex.py
======================

Fairness audit — Credit Risk Engine.

Responsibility
--------------
The SHAP analysis in explain_model.py showed SEX (encoded as
SEX_1.0 / SEX_2.0) ranking 6th of 19 in feature importance, with a
clear, systematic directional effect on the predicted default
probability. Using sex as a direct input to a credit decision is
prohibited or heavily restricted under most consumer-lending
fairness regulations (e.g. ECOA in the US, and equivalent
anti-discrimination law elsewhere), regardless of whether it is
statistically predictive.

This script answers the only question that matters before deciding
whether to remove it: how much predictive performance, if any, is
actually lost by excluding SEX from the model? If the AUC drop is
negligible, there is no performance justification for keeping a
protected attribute in a lending model.

This script does NOT retrain the full pipeline from scratch — it
reuses train_final.csv / test_final.csv (already encoded/scaled)
and simply drops the SEX_* columns before training, so the
comparison is a clean, isolated ablation test.

Pipeline position
------------------
    explain_model.py -> [fairness_check_sex.py] -> decision: retrain without SEX or keep + document rationale

Input
-----
    data/features/train_final.csv
    data/features/test_final.csv

Output
------
    models/artifacts/fairness_check_sex_report.json
"""

import os
import json
import logging
import pandas as pd
from xgboost import XGBClassifier

from src.utils.metrics import evaluate_classifier
from src.models.cross_validate_challenger import XGB_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "default payment_next_month"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
SEX_COLUMNS = ["SEX_1.0", "SEX_2.0"]  # ajusta si tu OneHotEncoder generó otros nombres


def load_splits():
    train = pd.read_csv(os.path.join(FEATURES_DIR, "train_final.csv"))
    test = pd.read_csv(os.path.join(FEATURES_DIR, "test_final.csv"))
    return train, test


def train_and_evaluate(train, test, drop_cols):
    X_train = train.drop(columns=[TARGET_COL] + drop_cols)
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL] + drop_cols)
    y_test = test[TARGET_COL]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return evaluate_classifier(y_test, y_pred, y_proba)


def run():
    train, test = load_splits()

    present_sex_cols = [c for c in SEX_COLUMNS if c in train.columns]
    if not present_sex_cols:
        logging.warning(f"No se encontraron columnas {SEX_COLUMNS} en train_final.csv. "
                         f"Revisa los nombres reales con: print(train.columns.tolist())")
        return None

    logging.info("Entrenando modelo CON SEX (referencia)...")
    with_sex_metrics = train_and_evaluate(train, test, drop_cols=[])

    logging.info(f"Entrenando modelo SIN {present_sex_cols}...")
    without_sex_metrics = train_and_evaluate(train, test, drop_cols=present_sex_cols)

    auc_with = with_sex_metrics["auc_roc"]
    auc_without = without_sex_metrics["auc_roc"]
    auc_delta = auc_with - auc_without

    logging.info(f"AUC-ROC CON sexo:    {auc_with:.4f}")
    logging.info(f"AUC-ROC SIN sexo:    {auc_without:.4f}")
    logging.info(f"Diferencia:          {auc_delta:+.4f}")

    if auc_delta < 0.005:
        logging.info("=> La perdida de AUC es minima (<0.005). "
                     "No hay justificacion de desempeno para mantener SEX en el modelo.")
    else:
        logging.info("=> La perdida de AUC es notoria. Se requiere una decision explicita "
                     "de negocio/legal sobre si el desempeno justifica el riesgo de compliance.")

    report = {
        "with_sex": with_sex_metrics,
        "without_sex": without_sex_metrics,
        "auc_delta": auc_delta,
        "columns_dropped": present_sex_cols,
    }

    report_path = os.path.join(ARTIFACTS_DIR, "fairness_check_sex_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return report


if __name__ == "__main__":
    run()
