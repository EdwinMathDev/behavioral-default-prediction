"""
test_api_v2.py
=================

API tests — Behavioral Default Prediction v2 (Home Credit).

Uses FastAPI's TestClient (no real server needed). Since RiskModelV2 needs
the trained v2 artifacts and feature store to load, these tests skip
gracefully (not fail) on a machine where the v2 pipeline hasn't been run
yet — same convention as the other v2 tests in this project.
"""

import os
import pytest

REQUIRED_ARTIFACTS = [
    os.path.join("config", "model_config_v2.json"),
    os.path.join("data", "features", "home_credit_features.csv"),
    os.path.join("data", "features", "train_final_v2.csv"),
    os.path.join("models", "artifacts", "encoder_v2.joblib"),
    os.path.join("models", "artifacts", "scaler_v2.joblib"),
]


def _artifacts_present():
    return all(os.path.exists(p) for p in REQUIRED_ARTIFACTS)


@pytest.fixture(scope="module")
def client():
    if not _artifacts_present():
        pytest.skip("v2 pipeline artifacts not found — run the v2 pipeline first (see README 'Running it — v2').")
    from fastapi.testclient import TestClient
    from src.api.main_v2 import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["known_clients"] > 0


def test_predict_unknown_client_returns_404(client):
    # SK_ID_CURR range in this dataset is roughly 100,000-460,000 — 1 is
    # guaranteed not to exist.
    response = client.get("/predict/1")
    assert response.status_code == 404


def test_predict_known_client_returns_valid_response(client):
    # Use the first client in the feature store, guaranteed to exist.
    import pandas as pd
    feature_store = pd.read_csv(os.path.join("data", "features", "home_credit_features.csv"))
    known_id = int(feature_store["SK_ID_CURR"].iloc[0])

    response = client.get(f"/predict/{known_id}")
    assert response.status_code == 200

    body = response.json()
    assert body["sk_id_curr"] == known_id
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["decision"] in ("approve", "reject")
    assert len(body["top_factors"]) == 5
    for factor in body["top_factors"]:
        assert factor["direction"] in ("increases_risk", "decreases_risk")


def test_predict_decision_matches_threshold(client):
    import pandas as pd
    feature_store = pd.read_csv(os.path.join("data", "features", "home_credit_features.csv"))
    known_id = int(feature_store["SK_ID_CURR"].iloc[0])

    response = client.get(f"/predict/{known_id}")
    body = response.json()

    expected_decision = "reject" if body["default_probability"] >= body["decision_threshold"] else "approve"
    assert body["decision"] == expected_decision
