"""
app_v2.py
===========

Streamlit dashboard — Behavioral Default Prediction v2 (Home Credit).

Responsibility
--------------
Same two-view structure as v1's dashboard, adapted for v2's client-lookup
model:

    1. "Scoring en vivo" — enter an existing SK_ID_CURR, call
       GET /predict/{sk_id_curr} on the running v2 API, display the
       decision, probability, and SHAP explanation.
    2. "Desempeño del modelo" — shows the saved evaluation metrics for
       whichever model is active per config/model_config_v2.json, plus
       the full promotion history (including the honest EXT_SOURCE
       finding) so a non-technical stakeholder can see the complete
       picture, not just a headline AUC number.

Requirement
-----------
The v2 API (src/api/main_v2.py) must be running separately:
    uvicorn src.api.main_v2:app --reload --port 8001

Run this dashboard with:
    streamlit run dashboard/app_v2.py --server.port 8502

(Port 8502, not 8501 — so v1's dashboard can run alongside it if needed.)
"""

import json
import os
import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8001"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
CONFIG_PATH = os.path.join("config", "model_config_v2.json")

with open(CONFIG_PATH) as f:
    config = json.load(f)

active_model_path = config["model"]["active_model_path"]
active_model_name = os.path.splitext(os.path.basename(active_model_path))[0]
active_threshold = config["decision_threshold"]["value"]

st.set_page_config(page_title="Behavioral Default Prediction v2", layout="wide")

st.title("🏠 Behavioral Default Prediction — v2 (Home Credit)")
st.caption(
    f"Active model: {active_model_name} — scores EXISTING clients by ID "
    f"(behavioral features come from a precomputed feature store, not a form)"
)

page = st.sidebar.radio("Vista", ["Scoring en vivo", "Desempeño del modelo"])

# ------------------------------------------------------------------
# Vista 1: Scoring en vivo
# ------------------------------------------------------------------
if page == "Scoring en vivo":
    st.header("Evaluar un cliente existente")

    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if not health.get("model_loaded"):
            st.error("The API responded but the model isn't loaded.")
        else:
            st.caption(f"Feature store: {health['known_clients']:,} known clients")
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the v2 API at http://127.0.0.1:8001. "
            "Make sure it's running: `uvicorn src.api.main_v2:app --reload --port 8001`"
        )
        st.stop()

    sk_id_curr = st.number_input(
        "SK_ID_CURR (client ID)",
        min_value=100000, max_value=500000, value=100002, step=1,
        help="Pick any SK_ID_CURR present in data/features/home_credit_features.csv "
             "(e.g. 100002, 100003, 100004 exist in the original Kaggle dataset).",
    )

    if st.button("Evaluar cliente", type="primary"):
        with st.spinner("Calculando score y explicación..."):
            response = requests.get(f"{API_URL}/predict/{sk_id_curr}", timeout=15)

        if response.status_code == 404:
            st.warning(f"Client {sk_id_curr} not found in the feature store. Try a different ID.")
        elif response.status_code != 200:
            st.error(f"API error: {response.text}")
        else:
            result = response.json()

            st.divider()
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                proba = result["default_probability"]
                decision = result["decision"]

                if decision == "reject":
                    st.error("### ❌ RECHAZAR")
                else:
                    st.success("### ✅ APROBAR")

                st.metric("Probabilidad de default", f"{proba:.1%}")
                st.caption(f"Threshold de decisión: {result['decision_threshold']:.1%}")
                st.caption(f"Modelo: {result['model_version']}")
                st.caption(f"Cliente: {result['sk_id_curr']}")

            with res_col2:
                st.subheader("Factores que más influyeron")
                factors_df = pd.DataFrame(result["top_factors"])
                factors_df["color"] = factors_df["direction"].map(
                    {"increases_risk": "Aumenta riesgo", "decreases_risk": "Reduce riesgo"}
                )
                st.bar_chart(factors_df.set_index("feature")["shap_value"])
                st.dataframe(
                    factors_df[["feature", "shap_value", "color"]].rename(
                        columns={"feature": "Variable", "shap_value": "Valor SHAP", "color": "Efecto"}
                    ),
                    hide_index=True,
                )

# ------------------------------------------------------------------
# Vista 2: Desempeño del modelo
# ------------------------------------------------------------------
else:
    st.header("Desempeño del modelo activo")
    st.caption(
        f"Mostrando métricas de **{active_model_name}** (modelo activo según "
        f"model_config_v2.json) con threshold de decisión = {active_threshold:.3f}"
    )

    metrics_path = os.path.join(ARTIFACTS_DIR, f"{active_model_name}_metrics.json")
    if not os.path.exists(metrics_path):
        st.warning(f"No se encontró {metrics_path}.")
        st.stop()

    with open(metrics_path) as f:
        metrics = json.load(f)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
    m2.metric("KS statistic", f"{metrics['ks_statistic']:.3f}")
    m3.metric("Precision", f"{metrics['precision']:.3f}")
    m4.metric("Recall", f"{metrics['recall']:.3f}")

    st.divider()

    fig_col1, fig_col2 = st.columns(2)
    roc_path = os.path.join(FIGURES_DIR, f"{active_model_name}_roc.png")
    cm_path = os.path.join(FIGURES_DIR, f"{active_model_name}_confusion_matrix.png")
    if os.path.exists(roc_path):
        fig_col1.image(roc_path, caption="Curva ROC")
    if os.path.exists(cm_path):
        fig_col2.image(cm_path, caption="Matriz de confusión")

    st.divider()
    st.subheader("Explicabilidad global (SHAP)")
    shap_path = os.path.join(FIGURES_DIR, f"{active_model_name}_shap_summary_plot.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="Impacto de cada variable en la predicción", width=700)
    else:
        st.info("Corre `python -m src.explainability.explain_model_v2` para generar este gráfico.")

    st.divider()
    st.subheader("Historial de selección de modelo y hallazgos honestos")
    st.text_area(
        "promotion_notes (config/model_config_v2.json)",
        value=config["model"]["promotion_notes"],
        height=300,
        disabled=True,
    )
