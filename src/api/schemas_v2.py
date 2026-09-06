"""
schemas_v2.py
===============

Request/response contracts for the v2 (Home Credit) API.

Responsibility
--------------
Unlike v1's schemas.py, there is no ApplicantData input model here. v2's
model depends on behavioral aggregates (PCT_INSTALLMENTS_LATE,
CC_AVG_UTILIZATION, bureau history, etc.) computed from 4 large auxiliary
tables — not something an applicant fills into a form, and not something
feasible to compute live per-request. In a real system, these come from a
feature store keyed by client ID. This API models that reality: it scores
an EXISTING client by SK_ID_CURR, looked up from the precomputed feature
table, rather than accepting raw applicant-submitted fields.

PredictionResponse and TopFactor are structurally identical to v1's —
duplicated here (not imported from v1) to keep the two API versions
independent, since they will eventually diverge further.
"""

from pydantic import BaseModel


class TopFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class PredictionResponse(BaseModel):
    sk_id_curr: int
    default_probability: float
    decision_threshold: float
    decision: str  # "reject" | "approve"
    top_factors: list[TopFactor]
    model_version: str
