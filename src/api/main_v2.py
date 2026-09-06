"""
main_v2.py
============

FastAPI application — Behavioral Default Prediction v2 (Home Credit).

Responsibility
--------------
Exposes the v2 model (whichever one config/model_config_v2.json currently
points to — XGBoost as of this writing, promoted after a paired t-test)
as an HTTP service. Unlike v1, this API scores EXISTING clients by
SK_ID_CURR — see inference_v2.py's docstring for why: the model's
behavioral features come from a precomputed feature store, not from
applicant-submitted form fields.

Endpoints
---------
    GET  /health              — liveness check
    GET  /predict/{sk_id_curr} — score an existing client by ID

Run locally
-----------
    uvicorn src.api.main_v2:app --reload --port 8001

(Port 8001, not 8000 — so v1's API can run alongside it if needed.)
Then open http://127.0.0.1:8001/docs for interactive API docs.
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException

from src.api.schemas_v2 import PredictionResponse
from src.api.inference_v2 import RiskModelV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Behavioral Default Prediction API (v2 — Home Credit)",
    description="Scores an EXISTING client (by SK_ID_CURR) using behavioral "
                 "features aggregated across 5 relational tables (bureau, "
                 "previous applications, installment payments, credit card "
                 "balances). Active model and threshold read from "
                 "config/model_config_v2.json — see that file's promotion_notes "
                 "for the full model-selection and fairness history, including "
                 "an honest accounting of how much of this model's performance "
                 "traces to precomputed external scores rather than this "
                 "project's own feature engineering.",
    version="2.0.0",
)

risk_model: Optional[RiskModelV2] = None


@app.on_event("startup")
def load_model():
    global risk_model
    risk_model = RiskModelV2()
    logging.info("RiskModelV2 loaded and ready to receive requests.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": risk_model is not None,
        "known_clients": len(risk_model.feature_store) if risk_model else 0,
    }


@app.get("/predict/{sk_id_curr}", response_model=PredictionResponse)
def predict(sk_id_curr: int):
    if risk_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    if not risk_model.client_exists(sk_id_curr):
        raise HTTPException(
            status_code=404,
            detail=f"SK_ID_CURR {sk_id_curr} not found in the feature store. "
                    f"This API only scores existing clients — see /health for "
                    f"the count of known clients, or pick a valid ID from "
                    f"data/features/home_credit_features.csv.",
        )

    try:
        result = risk_model.predict(sk_id_curr)
    except Exception as e:
        logging.error(f"Error processing request for {sk_id_curr}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {e}")

    return result
