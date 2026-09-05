"""
test_config_v2.py
====================

Config sanity tests — Home Credit Default Risk (v2).

Mirrors v1's test_config.py, adapted for config/model_config_v2.json's
slightly different structure (adds oof_predictions_path, and
decision_threshold.value may legitimately be null before
optimize_threshold_v2_active.py has been run at least once).
"""

import json
import os

import pytest

CONFIG_PATH = os.path.join("config", "model_config_v2.json")


@pytest.fixture
def config():
    if not os.path.exists(CONFIG_PATH):
        pytest.skip(f"{CONFIG_PATH} not found.")
    with open(CONFIG_PATH) as f:
        return json.load(f)


def test_config_file_loads_without_error(config):
    assert isinstance(config, dict)


def test_config_has_required_top_level_keys(config):
    required = {"model", "decision_threshold", "business_cost_assumptions", "target_column", "oof_predictions_path"}
    assert required.issubset(config.keys())


def test_active_model_path_is_a_string(config):
    assert isinstance(config["model"]["active_model_path"], str)
    assert config["model"]["active_model_path"].endswith(".joblib")


def test_active_model_file_exists_on_disk(config):
    model_path = config["model"]["active_model_path"]
    if not os.path.exists(model_path):
        pytest.skip(f"{model_path} not found — model hasn't been trained on this machine yet.")
    assert os.path.exists(model_path)


def test_decision_threshold_is_valid_probability_when_set(config):
    threshold = config["decision_threshold"]["value"]
    if threshold is None:
        pytest.skip("decision_threshold.value is null — optimize_threshold_v2_active.py hasn't been run yet.")
    assert 0.0 <= threshold <= 1.0


def test_target_column_matches_home_credit_dataset(config):
    assert config["target_column"] == "TARGET"


def test_promotion_notes_is_non_empty(config):
    notes = config["model"].get("promotion_notes", "")
    assert len(notes) > 0, "promotion_notes should document why the active model was chosen"
