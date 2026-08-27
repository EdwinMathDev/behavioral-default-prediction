"""
test_build_features.py
=======================

Tests for src/features/build_features.py.

Why these tests exist
----------------------
- test_sex_not_in_base_columns: SEX was removed from the model after
  a fairness finding (see FAIRNESS.md). This is a regression guard —
  if someone adds SEX back to BASE_COLUMNS in the future (by mistake,
  or copy-pasting from an old version), this test fails immediately
  instead of the issue being rediscovered months later via SHAP.
- test_ratio_clipping: payment_ratio/utilization_ratio had unbounded
  values in an early version of this project, before clipping was
  added. This test guards against that regressing silently.
- test_no_inf_or_extreme_nan: build_features.py generates ratios via
  division; this confirms the inf-sanitization step still works.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import create_features, BASE_COLUMNS, RATIO_CLIP_MAX


@pytest.fixture
def raw_applicant_df():
    """Un DataFrame mínimo con las columnas crudas esperadas, para
    varios solicitantes con distintos perfiles de riesgo."""
    return pd.DataFrame({
        "LIMIT_BAL": [200000, 50000, 10000],
        "EDUCATION": [2, 1, 3],
        "MARRIAGE": [1, 2, 1],
        "AGE": [34, 45, 23],
        "PAY_0": [0, -1, 3],
        "PAY_2": [0, -1, 2],
        "PAY_3": [0, -1, 2],
        "PAY_4": [0, -1, 1],
        "PAY_5": [0, -1, 1],
        "PAY_6": [0, -1, 0],
        "BILL_AMT1": [50000, 1000, 9000],
        "BILL_AMT2": [48000, 900, 8500],
        "BILL_AMT3": [45000, 800, 8000],
        "BILL_AMT4": [42000, 700, 7500],
        "BILL_AMT5": [40000, 600, 7000],
        "BILL_AMT6": [38000, 500, 6500],
        "PAY_AMT1": [3000, 900, 100],
        "PAY_AMT2": [3000, 800, 50],
        "PAY_AMT3": [3000, 700, 0],
        "PAY_AMT4": [3000, 600, 0],
        "PAY_AMT5": [3000, 500, 0],
        "PAY_AMT6": [3000, 400, 0],
        "default payment_next_month": [0, 0, 1],
    })


def test_sex_not_in_base_columns():
    """SEX fue removida tras el hallazgo de fairness (ver FAIRNESS.md).
    Si esto falla, alguien la reintrodujo por error."""
    assert "SEX" not in BASE_COLUMNS


def test_create_features_excludes_sex(raw_applicant_df):
    result = create_features(raw_applicant_df)
    sex_cols = [c for c in result.columns if c.upper().startswith("SEX")]
    assert sex_cols == [], f"Columnas de SEX encontradas en el output: {sex_cols}"


def test_ratio_columns_are_clipped(raw_applicant_df):
    result = create_features(raw_applicant_df)

    for i in range(1, 7):
        col = f"utilization_ratio_{i}"
        assert result[col].between(0, 2).all(), f"{col} fuera del rango esperado [0, 2]"

    for i in range(1, 7):
        col = f"payment_ratio_{i}"
        assert result[col].between(0, RATIO_CLIP_MAX).all(), f"{col} fuera del rango esperado [0, {RATIO_CLIP_MAX}]"


def test_no_inf_values(raw_applicant_df):
    result = create_features(raw_applicant_df)
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    assert not np.isinf(result[numeric_cols]).any().any(), "Se encontraron valores infinitos en el output"


def test_output_row_count_matches_input(raw_applicant_df):
    result = create_features(raw_applicant_df)
    assert len(result) == len(raw_applicant_df)
