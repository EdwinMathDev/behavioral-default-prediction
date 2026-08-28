"""
inference.py
=============

Inference pipeline for a single applicant — Credit Risk Engine API.

Responsibility
--------------
Applies EXACTLY the same transformation sequence used at training
time (build_features -> encode -> log1p -> impute -> scale) to a
single new applicant, using the persisted artifacts (encoder,
scaler, imputation medians, feature column order) rather than
refitting anything. This is the guarantee that a prediction served
in production matches what the model actually learned.

All artifacts and the active model/threshold are read from
config/model_config.json — this module never hardcodes a model path
or threshold value.
"""

import os
import logging
import joblib
import shap
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.features.build_features import create_features, PAY_DELAY_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARTIFACTS_DIR = "models/artifacts"
CATEGORICAL_COLS = ["EDUCATION", "MARRIAGE", "age_group"]


class RiskModel:
    """Carga todos los artefactos una sola vez al iniciar la API
    (no en cada request) y expone un método predict() simple."""

    def __init__(self):
        config = load_config()
        self.threshold = config["decision_threshold"]["value"]
        self.target_col = config["target_column"]

        model_path = config["model"]["active_model_path"]
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(config["model"]["encoder_path"])
        self.scaler = joblib.load(config["model"]["scaler_path"])
        self.medians = joblib.load(os.path.join(ARTIFACTS_DIR, "impute_medians.joblib"))
        self.feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))

        # TreeExplainer se instancia una sola vez — es costoso crearlo
        # por request, pero explicar UNA fila con uno ya creado es rápido.
        self.explainer = shap.TreeExplainer(self.model)

        logging.info(f"Modelo cargado: {model_path}")
        logging.info(f"Threshold activo: {self.threshold}")

    def _transform(self, applicant_dict: dict) -> pd.DataFrame:
        # 1. Arma un DataFrame de una fila con las columnas crudas.
        #    create_features() espera también la columna target — se
        #    agrega como dummy (no se usa para calcular features).
        row = applicant_dict.copy()
        row[self.target_col] = 0
        df = pd.DataFrame([row])

        # 2. Mismo feature engineering que en entrenamiento.
        df = create_features(df)
        df = df.drop(columns=[self.target_col])

        # 3. Encoding categórico — usa el encoder YA AJUSTADO en train.
        encoded = pd.DataFrame(
            self.encoder.transform(df[CATEGORICAL_COLS]),
            columns=self.encoder.get_feature_names_out(CATEGORICAL_COLS),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=CATEGORICAL_COLS), encoded], axis=1)

        # 4. log1p en PAY_AMT — mismo transform determinístico de train.
        pay_amt_cols = [c for c in df.columns if "PAY_AMT" in c]
        for col in pay_amt_cols:
            df[col] = np.log1p(df[col].clip(lower=0))

        # 5. Imputar con las medianas de TRAIN (no las de este único caso).
        df = df.fillna(self.medians)

        # 6. Reindexar a las columnas EXACTAS que el modelo espera, en el
        #    mismo orden — cualquier columna faltante (ej. una categoría
        #    de EDUCATION que no aparece en este request) se llena con 0.
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # 7. Escalar con el scaler YA AJUSTADO en train.
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    def predict(self, applicant_dict: dict) -> dict:
        X = self._transform(applicant_dict)

        proba = float(self.model.predict_proba(X)[:, 1][0])
        decision = "reject" if proba >= self.threshold else "approve"

        shap_values = self.explainer(X)
        contributions = pd.Series(shap_values.values[0], index=X.columns)
        top = contributions.abs().sort_values(ascending=False).head(5)

        top_factors = [
            {
                "feature": feat,
                "shap_value": float(contributions[feat]),
                "direction": "increases_risk" if contributions[feat] > 0 else "decreases_risk",
            }
            for feat in top.index
        ]

        return {
            "default_probability": round(proba, 4),
            "decision_threshold": self.threshold,
            "decision": decision,
            "top_factors": top_factors,
            "model_version": os.path.basename(load_config()["model"]["active_model_path"]),
        }
