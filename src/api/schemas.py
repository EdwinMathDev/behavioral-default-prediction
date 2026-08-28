"""
schemas.py
==========

Request/response contracts for the Credit Risk Engine API.

Responsibility
--------------
Defines the exact shape of data the API accepts and returns.
ApplicantData mirrors the raw applicant fields from the original
dataset (SEX intentionally excluded — see FAIRNESS.md). Pydantic
validates types and ranges before any of this data reaches the
model pipeline.
"""

from pydantic import BaseModel, Field, ConfigDict


class ApplicantData(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Límite de crédito otorgado (NT$)")
    EDUCATION: int = Field(..., ge=0, le=6, description="1=posgrado, 2=universidad, 3=preparatoria, 4=otros")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1=casado, 2=soltero, 3=otros")
    AGE: int = Field(..., ge=18, le=100)

    PAY_0: int = Field(..., description="Estado de pago mes más reciente (-1=al día, 1+=meses de atraso)")
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "LIMIT_BAL": 200000, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 34,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 45000,
            "BILL_AMT4": 42000, "BILL_AMT5": 40000, "BILL_AMT6": 38000,
            "PAY_AMT1": 3000, "PAY_AMT2": 3000, "PAY_AMT3": 3000,
            "PAY_AMT4": 3000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
        }
    })


class TopFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class PredictionResponse(BaseModel):
    default_probability: float
    decision_threshold: float
    decision: str  # "reject" | "approve"
    top_factors: list[TopFactor]
    model_version: str
