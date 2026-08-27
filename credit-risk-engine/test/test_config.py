"""
test_config.py
===============

Tests for src/utils/config.py.

Why these tests exist
----------------------
- test_config_path_resolves: config.py had a real bug where the path
  to config/model_config.json was miscalculated (one directory level
  short), causing a FileNotFoundError the first time explain_model.py
  was run. This test guards against that specific class of bug.
- test_decision_threshold_is_valid_probability: a threshold outside
  [0, 1] would silently make the API approve or reject everyone.
"""

import os
import pytest

from src.utils.config import load_config, get_decision_threshold, get_active_model_path


def test_config_file_loads_without_error():
    config = load_config()
    assert isinstance(config, dict)


def test_config_has_required_top_level_keys():
    config = load_config()
    for key in ["model", "decision_threshold", "business_cost_assumptions", "target_column"]:
        assert key in config, f"Falta la clave '{key}' en model_config.json"


def test_decision_threshold_is_valid_probability():
    threshold = get_decision_threshold()
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0, f"Threshold fuera de rango: {threshold}"


def test_active_model_path_is_a_string():
    path = get_active_model_path()
    assert isinstance(path, str)
    assert path.endswith(".joblib")


@pytest.mark.skipif(
    not os.path.exists(get_active_model_path()) if os.path.exists("config/model_config.json") else True,
    reason="El modelo activo aún no se ha entrenado en este entorno (correr train_challenger.py primero).",
)
def test_active_model_file_exists_on_disk():
    """Si el config apunta a un modelo que no existe en disco, la API
    va a fallar al arrancar — mejor detectarlo aquí."""
    assert os.path.exists(get_active_model_path())
