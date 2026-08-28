"""
main.py
=======

FastAPI application — Credit Risk Engine.

Responsibility
--------------
Exposes the trained, fairness-audited XGBoost model as an HTTP
service. The RiskModel (inference.py) is loaded ONCE at startup,
not per-request — loading a model and re-fitting an explainer on
every request would be both slow and wasteful.

Endpoints
---------
    GET  /health    — liveness check
    POST /predict    — score a single applicant

Run locally
-----------
    uvicorn src.api.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

import logging
from typing import Optional
from fastapi import FastAPI, HTTPException

from src.api.schemas import ApplicantData, PredictionResponse
from src.api.inference import RiskModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Credit Risk Engine API",
    description="Scores credit-default risk for new applicants using an "
                 "XGBoost model, audited for fairness (see FAIRNESS.md).",
    version="1.0.0",
)

# Se carga una sola vez, al iniciar el servidor — no en cada request.
risk_model: Optional[RiskModel] = None


@app.on_event("startup")
def load_model():
    global risk_model
    risk_model = RiskModel()
    logging.info("RiskModel cargado y listo para recibir requests.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": risk_model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantData):
    if risk_model is None:
        raise HTTPException(status_code=503, detail="Modelo aún no cargado. Intenta de nuevo en un momento.")

    try:
        result = risk_model.predict(applicant.model_dump())
    except Exception as e:
        logging.error(f"Error al procesar la solicitud: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {e}")

    return result
