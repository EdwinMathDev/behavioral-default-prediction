"""
test_fairness_no_protected_attributes_v2.py
==============================================

Fairness regression tests — Home Credit Default Risk (v2).

Mirrors v1's test_fairness_no_protected_attributes.py: guards against
CODE_GENDER silently re-entering the v2 pipeline after the ablation
documented in config/model_config_v2.json (negligible AUC cost, +0.0018,
confirmed via 5-fold CV) led to its removal.

If any of these tests fail after a future code change, that change
re-introduced a protected attribute into the v2 pipeline and needs an
explicit, evidence-based fairness decision before merging — the same bar
already applied to SEX in v1 and CODE_GENDER here.
"""

import json
import os

import joblib
import pytest

PROTECTED_COLUMNS_V2 = ["CODE_GENDER"]


@pytest.mark.parametrize("protected_col", PROTECTED_COLUMNS_V2)
def test_protected_column_not_in_application_base_columns(protected_col):
    from src.features.build_features_v2 import APPLICATION_BASE_COLUMNS
    assert protected_col not in APPLICATION_BASE_COLUMNS


@pytest.mark.parametrize("protected_col", PROTECTED_COLUMNS_V2)
def test_protected_column_not_in_train_pipeline_categorical_cols(protected_col):
    from src.models.train_pipeline_v2 import CATEGORICAL_COLS
    assert protected_col not in CATEGORICAL_COLS


@pytest.mark.parametrize("protected_col", PROTECTED_COLUMNS_V2)
def test_protected_column_not_in_cross_validate_categorical_cols(protected_col):
    from src.models.cross_validate_baseline_v2 import CATEGORICAL_COLS
    assert protected_col not in CATEGORICAL_COLS


def test_trained_model_feature_columns_exclude_gender():
    """If the v2 model has been trained at least once, its persisted
    feature-columns artifact must not contain any CODE_GENDER_* column."""
    feature_columns_path = os.path.join("models", "artifacts", "feature_columns_v2.joblib")
    if not os.path.exists(feature_columns_path):
        pytest.skip("feature_columns_v2.joblib not found — run train_pipeline_v2.py first.")

    feature_columns = joblib.load(feature_columns_path)
    gender_cols = [c for c in feature_columns if c.startswith("CODE_GENDER")]
    assert gender_cols == [], f"Found gender-derived columns in the trained v2 model: {gender_cols}"


def test_model_config_v2_documents_the_gender_decision():
    """The exclusion of CODE_GENDER must be documented, not just
    implemented silently — same principle as v1's FAIRNESS.md."""
    config_path = os.path.join("config", "model_config_v2.json")
    if not os.path.exists(config_path):
        pytest.skip("config/model_config_v2.json not found.")

    with open(config_path) as f:
        config = json.load(f)

    notes = config.get("model", {}).get("promotion_notes", "")
    assert "CODE_GENDER" in notes, (
        "model_config_v2.json's promotion_notes should document the CODE_GENDER "
        "fairness decision, same as it documents the model promotion history."
    )
