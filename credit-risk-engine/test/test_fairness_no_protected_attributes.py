"""
test_fairness_no_protected_attributes.py
==========================================

Fairness regression tests — Credit Risk Engine.

Why these tests exist
----------------------
SHAP analysis found SEX ranking 6th of 19 features by importance,
with a systematic effect on predicted default risk (see FAIRNESS.md).
It was removed from every stage of the pipeline. Nothing besides
these tests currently prevents someone from reintroducing it by
accident — e.g. copy-pasting an older version of a file, or adding a
new categorical column without checking this list.

If any of these tests fail, treat it as a blocking issue: it means a
protected attribute may be back in the model, which is both a
compliance and an ethical problem, not just a code smell.

PROTECTED_COLUMNS is deliberately explicit and small — extend it if
the project later determines other columns (e.g. AGE, in some
jurisdictions) should also be excluded or restricted. See the
"Follow-up" section of FAIRNESS.md for the current stance on AGE.
"""

import os
import joblib
import pytest

from src.data.preprocess import CATEGORICAL_COLS as PREPROCESS_CATEGORICAL_COLS
from src.features.build_features import BASE_COLUMNS
from src.models.train_pipeline import CATEGORICAL_COLS as PIPELINE_CATEGORICAL_COLS

PROTECTED_COLUMNS = ["SEX"]
ARTIFACTS_DIR = "models/artifacts"


@pytest.mark.parametrize("protected", PROTECTED_COLUMNS)
def test_protected_column_not_in_base_columns(protected):
    assert protected not in BASE_COLUMNS


@pytest.mark.parametrize("protected", PROTECTED_COLUMNS)
def test_protected_column_not_in_preprocess_categorical_cols(protected):
    assert protected not in PREPROCESS_CATEGORICAL_COLS


@pytest.mark.parametrize("protected", PROTECTED_COLUMNS)
def test_protected_column_not_in_train_pipeline_categorical_cols(protected):
    assert protected not in PIPELINE_CATEGORICAL_COLS


@pytest.mark.skipif(
    not os.path.exists(os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")),
    reason="feature_columns.joblib aún no existe — correr train_pipeline.py primero.",
)
def test_trained_model_feature_columns_exclude_protected_attributes():
    """El test más importante de este archivo: revisa el artefacto REAL
    que el modelo entrenado usa, no solo el código fuente. Si alguien
    reentrena con una versión vieja de un archivo, esto lo detecta."""
    feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))
    for protected in PROTECTED_COLUMNS:
        matches = [c for c in feature_columns if c.upper().startswith(protected)]
        assert matches == [], (
            f"Columnas relacionadas con el atributo protegido '{protected}' "
            f"encontradas en el modelo entrenado: {matches}"
        )
