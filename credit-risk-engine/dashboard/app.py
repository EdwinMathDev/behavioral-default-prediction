"""
app.py
======

Streamlit dashboard — Credit Risk Engine.

Responsibility
--------------
Provides a visual, non-technical interface on top of the API
(src/api/main.py). Has two views:

    1. "Scoring en vivo" — a form to score a new applicant, calling
       POST /predict on the running API and displaying the decision,
       probability, and SHAP explanation.
    2. "Desempeño del modelo" — shows the saved evaluation metrics
       and figures (ROC, confusion matrix, SHAP summary) generated
       during training, so a non-technical stakeholder can see how
       the active model performs without digging through JSON files.

Requirement
-----------
The API (src/api/main.py) must be running separately:
    uvicorn src.api.main:app --reload --port 8000

Run this dashboard with:
    streamlit run dashboard/app.py
"""

import json
import os
import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
API_HEALTH_URL = "http://127.0.0.1:8000/health"
ARTIFACTS_DIR = "models/artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")

st.set_page_config(page_title="Credit Risk Engine", layout="wide")

st.title("💳 Credit Risk Engine")
st.caption("Modelo XGBoost auditado por fairness — ver FAIRNESS.md")

page = st.sidebar.radio("Vista", ["Scoring en vivo", "Desempeño del modelo"])

# ------------------------------------------------------------------
# Vista 1: Scoring en vivo
# ------------------------------------------------------------------
if page == "Scoring en vivo":
    st.header("Evaluar un solicitante nuevo")

    try:
        health = requests.get(API_HEALTH_URL, timeout=2).json()
        if not health.get("model_loaded"):
            st.error("La API respondió pero el modelo no está cargado.")
    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con la API en http://127.0.0.1:8000. "
            "Verifica que esté corriendo: `uvicorn src.api.main:app --reload --port 8000`"
        )
        st.stop()

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Datos generales")
            limit_bal = st.number_input("Límite de crédito (LIMIT_BAL)", min_value=1000, value=200000, step=10000)
            education = st.selectbox("Educación (EDUCATION)", [1, 2, 3, 4],
                                      format_func=lambda x: {1: "Posgrado", 2: "Universidad",
                                                              3: "Preparatoria", 4: "Otros"}[x])
            marriage = st.selectbox("Estado civil (MARRIAGE)", [1, 2, 3],
                                     format_func=lambda x: {1: "Casado", 2: "Soltero", 3: "Otros"}[x])
            age = st.number_input("Edad (AGE)", min_value=18, max_value=100, value=34)

        with col2:
            st.subheader("Historial de atraso (PAY_0 a PAY_6)")
            st.caption("-1 = al día · 0 = pago mínimo · 1+ = meses de atraso")
            pay_0 = st.number_input("PAY_0 (más reciente)", min_value=-2, max_value=9, value=0)
            pay_2 = st.number_input("PAY_2", min_value=-2, max_value=9, value=0)
            pay_3 = st.number_input("PAY_3", min_value=-2, max_value=9, value=0)
            pay_4 = st.number_input("PAY_4", min_value=-2, max_value=9, value=0)
            pay_5 = st.number_input("PAY_5", min_value=-2, max_value=9, value=0)
            pay_6 = st.number_input("PAY_6", min_value=-2, max_value=9, value=0)

        with col3:
            st.subheader("Facturación y pagos (últimos 6 meses)")
            bill_amts = [st.number_input(f"BILL_AMT{i}", value=40000, step=1000) for i in range(1, 7)]
            pay_amts = [st.number_input(f"PAY_AMT{i}", min_value=0, value=3000, step=500) for i in range(1, 7)]

        submitted = st.form_submit_button("Evaluar solicitante", type="primary")

    if submitted:
        payload = {
            "LIMIT_BAL": limit_bal, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            **{f"BILL_AMT{i+1}": bill_amts[i] for i in range(6)},
            **{f"PAY_AMT{i+1}": pay_amts[i] for i in range(6)},
        }

        with st.spinner("Calculando score y explicación..."):
            response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code != 200:
            st.error(f"Error de la API: {response.text}")
        else:
            result = response.json()

            st.divider()
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                proba = result["default_probability"]
                decision = result["decision"]

                if decision == "reject":
                    st.error(f"### ❌ RECHAZAR")
                else:
                    st.success(f"### ✅ APROBAR")

                st.metric("Probabilidad de default", f"{proba:.1%}")
                st.caption(f"Threshold de decisión: {result['decision_threshold']:.1%}")
                st.caption(f"Modelo: {result['model_version']}")

            with res_col2:
                st.subheader("Factores que más influyeron")
                factors_df = pd.DataFrame(result["top_factors"])
                factors_df["impacto"] = factors_df["shap_value"].abs()
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

    metrics_path = os.path.join(ARTIFACTS_DIR, "xgb_challenger_metrics.json")
    if not os.path.exists(metrics_path):
        st.warning(f"No se encontró {metrics_path}. Corre `python -m src.models.train_challenger` primero.")
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
    roc_path = os.path.join(FIGURES_DIR, "xgb_challenger_roc.png")
    cm_path = os.path.join(FIGURES_DIR, "xgb_challenger_confusion_matrix.png")
    if os.path.exists(roc_path):
        fig_col1.image(roc_path, caption="Curva ROC")
    if os.path.exists(cm_path):
        fig_col2.image(cm_path, caption="Matriz de confusión")

    st.divider()
    st.subheader("Explicabilidad global (SHAP)")
    shap_path = os.path.join(FIGURES_DIR, "shap_summary_plot.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="Impacto de cada variable en la predicción", width=700)
    else:
        st.info("Corre `python -m src.explainability.explain_model` para generar este gráfico.")

    st.divider()
    st.subheader("Nota de fairness")
    st.info(
        "La variable SEX fue removida del modelo tras un hallazgo de fairness en el "
        "análisis de SHAP (impacto sistemático en la predicción sin aporte real de "
        "desempeño). Ver FAIRNESS.md para el detalle completo del hallazgo y la decisión."
    )
