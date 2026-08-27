"""
test_api_schema.py
===================

Tests for src/api/schemas.py.

Why these tests exist
----------------------
- test_sex_not_a_field: the API contract must not accept SEX as
  input at all — not just drop it internally. If a consumer of this
  API could send SEX and have it silently ignored, that's a worse
  failure mode than rejecting it outright, since a caller might
  believe it's being used.
- test_valid_payload_accepted / test_invalid_education_rejected:
  basic contract tests so a future change to schemas.py that breaks
  validation is caught here, not in production.
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import ApplicantData

VALID_PAYLOAD = {
    "LIMIT_BAL": 200000, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 34,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 50000, "BILL_AMT2": 48000, "BILL_AMT3": 45000,
    "BILL_AMT4": 42000, "BILL_AMT5": 40000, "BILL_AMT6": 38000,
    "PAY_AMT1": 3000, "PAY_AMT2": 3000, "PAY_AMT3": 3000,
    "PAY_AMT4": 3000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
}


def test_sex_not_a_field():
    assert "SEX" not in ApplicantData.model_fields


def test_valid_payload_is_accepted():
    applicant = ApplicantData(**VALID_PAYLOAD)
    assert applicant.LIMIT_BAL == 200000


def test_invalid_education_is_rejected():
    bad_payload = {**VALID_PAYLOAD, "EDUCATION": 99}
    with pytest.raises(ValidationError):
        ApplicantData(**bad_payload)


def test_negative_limit_bal_is_rejected():
    bad_payload = {**VALID_PAYLOAD, "LIMIT_BAL": -500}
    with pytest.raises(ValidationError):
        ApplicantData(**bad_payload)


def test_underage_applicant_is_rejected():
    bad_payload = {**VALID_PAYLOAD, "AGE": 15}
    with pytest.raises(ValidationError):
        ApplicantData(**bad_payload)


def test_missing_required_field_is_rejected():
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "LIMIT_BAL"}
    with pytest.raises(ValidationError):
        ApplicantData(**incomplete_payload)
