"""
config.py
=========

Central configuration loader — Credit Risk Engine.

Responsibility
--------------
Provides a single access point to config/model_config.json — the
project's single source of truth for the active model path, the
chosen decision threshold, and the business cost assumptions used
to select it.

Any module that needs the decision threshold (evaluation scripts,
the future serving API, monitoring dashboards) must read it from
here, never hardcode it. This guarantees that a threshold update
made after re-optimization is picked up everywhere at once, and
that training-time and inference-time decisions never drift apart.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "model_config.json")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_decision_threshold() -> float:
    return load_config()["decision_threshold"]["value"]


def get_active_model_path() -> str:
    return load_config()["model"]["active_model_path"]


def get_target_column() -> str:
    return load_config()["target_column"]