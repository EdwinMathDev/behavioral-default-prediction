"""
inference_v2.py
==================

Inference pipeline for an existing client — Home Credit Default Risk (v2) API.

Responsibility
--------------
Scores a client identified by SK_ID_CURR, looked up from the precomputed
feature table (data/features/home_credit_features.csv) — simulating a
feature store lookup, since v2's behavioral features (bureau history,
installment lateness, credit card utilization) are aggregates over 4
large auxiliary tables that cannot realistically be computed live from a
request body.

Applies EXACTLY the same transformation sequence used at training time
(fill missing categoricals -> encode -> log1p -> impute -> scale) using
the persisted artifacts from train_pipeline_v2.py, so a prediction served
here matches what the model actually learned.

All artifacts and the active model/threshold are read from
config/model_config_v2.json — this module never hardcodes a model path
or threshold value.
"""

import os
import json
import logging
import joblib
import shap
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CONFIG_PATH = os.path.join("config", "model_config_v2.json")
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]
SKEWED_COLS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "BUREAU_TOTAL_DEBT"]
BACKGROUND_SAMPLE_SIZE = 100


class RiskModelV2:
    """Loads all artifacts once at startup (not per-request) and exposes
    a simple predict(sk_id_curr) method."""

    def __init__(self):
        with open(CONFIG_PATH) as f:
            config = json.load(f)

        self.threshold = config["decision_threshold"]["value"]
        self.target_col = config["target_column"]
        model_path = config["model"]["active_model_path"]
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        self.model = joblib.load(model_path)
        self.encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "encoder_v2.joblib"))
        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler_v2.joblib"))
        self.medians = joblib.load(os.path.join(ARTIFACTS_DIR, "impute_medians_v2.joblib"))
        self.feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns_v2.joblib"))

        # The precomputed "feature store" — one row per known client,
        # already aggregated across all 5 raw tables by build_features_v2.py.
        self.feature_store = pd.read_csv(
            os.path.join(FEATURES_DIR, "home_credit_features.csv")
        ).set_index("SK_ID_CURR")

        background_pool = pd.read_csv(
            os.path.join(FEATURES_DIR, "train_final_v2.csv")
        ).drop(columns=[self.target_col])
        sample_size = min(BACKGROUND_SAMPLE_SIZE, len(background_pool))
        background = background_pool.sample(n=sample_size, random_state=42)
        self.explainer = shap.Explainer(self.model, background)

        logging.info(f"Model loaded: {model_path}")
        logging.info(f"Active threshold: {self.threshold}")
        logging.info(f"Feature store loaded: {len(self.feature_store):,} known clients")
        logging.info(f"SHAP explainer: {type(self.explainer).__name__}")

    def client_exists(self, sk_id_curr: int) -> bool:
        return sk_id_curr in self.feature_store.index

    def _transform(self, raw_row: pd.Series) -> pd.DataFrame:
        row = raw_row.drop(labels=[self.target_col], errors="ignore")
        df = pd.DataFrame([row])

        present_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
        df[present_cat] = df[present_cat].fillna("Missing")

        encoded = pd.DataFrame(
            self.encoder.transform(df[present_cat]),
            columns=self.encoder.get_feature_names_out(present_cat),
            index=df.index,
        )
        df = pd.concat([df.drop(columns=present_cat), encoded], axis=1)

        present_skewed = [c for c in SKEWED_COLS if c in df.columns]
        for col in present_skewed:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

        df = df.reindex(columns=self.feature_columns, fill_value=0)
        df = df.fillna(self.medians)

        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    def predict(self, sk_id_curr: int) -> dict:
        if not self.client_exists(sk_id_curr):
            raise KeyError(f"SK_ID_CURR {sk_id_curr} not found in the feature store.")

        raw_row = self.feature_store.loc[sk_id_curr]
        X = self._transform(raw_row)

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
            "sk_id_curr": sk_id_curr,
            "default_probability": round(proba, 4),
            "decision_threshold": self.threshold,
            "decision": decision,
            "top_factors": top_factors,
            "model_version": self.model_name,
        }
