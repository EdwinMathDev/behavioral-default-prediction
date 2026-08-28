# Credit Risk Engine

A credit-default scoring system built around a simple premise: a model
is not finished when it achieves a good AUC. It is finished when its
predictions are reproducible, its decisions are explainable, its
threshold reflects the economics of the business it serves, and its
use of applicant data has been examined  not merely assumed  to be
fair.

This repository documents that process end to end, including the
points where the first version of the pipeline was wrong.

---

## What this is

A model that predicts the probability that a credit-card holder will
default on payment in the following month, trained on the UCI *Default
of Credit Card Clients* dataset, and served through an API and a
lightweight dashboard.

| Metric / Feature | Value / Details |
|---|---|
| **Active model** | XGBoost (gradient-boosted trees) |
| **AUC-ROC** | 0.768 |
| **KS statistic** | 0.402 |
| **Decision threshold** | 0.25 (cost-optimal, not the default 0.5) |
| **Fairness** | Audited with SHAP; sex excluded from the model |




## Why XGBoost, and not just "the model that scored highest"

A Logistic Regression baseline was trained first, deliberately  in
consumer lending, an interpretable linear model is the right starting
point, not an afterthought. XGBoost was only adopted after it beat the
baseline consistently across 5-fold cross-validation (a gain in 4 of 5
folds, not just a better average), and after both models were compared
at *their own* cost-optimal decision threshold rather than the
arbitrary default of 0.5.

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| Decision threshold | 0.44 | 0.25 |
| AUC-ROC | 0.752 | 0.768 |
| Recall (defaults caught) | 66.8% | 80.8% |
| False negatives (missed defaults) | 440 | 255 |

The gain was judged against the baseline's own fold-to-fold variance,
not against zero  a improvement smaller than a model's natural noise
is not an improvement.

## A finding worth stating plainly

SHAP analysis of the trained model showed `SEX` ranking 6th of 19
features by importance, with a consistent, systematic effect on
predicted risk  not statistical noise. An ablation test confirmed
what that implies: removing it cost 0.0011 AUC, roughly fifteen times
smaller than the margin that justified choosing XGBoost over the
baseline in the first place. There was no predictive case for keeping
it, and a clear compliance and fairness case for removing it. It was
removed from every stage of the pipeline, the finding is recorded in
[`FAIRNESS.md`](FAIRNESS.md), and a test suite now guards against its
silent reintroduction (`test/test_fairness_no_protected_attributes.py`).

This is included in the README, rather than left to a footnote,
because it is the decision in this project most worth being
transparent about.

---

## Architecture

```
raw data
   │
   ▼
preprocess.py            cleaning: imputation, category correction
   │
   ▼
build_features.py        domain features: utilization, payment ratios,
   │                     delinquency history, trend, volatility
   ▼
train_pipeline.py        stratified split → encode → scale → SMOTE
   │                     (all fit on train only — no leakage)
   ▼
train_challenger.py      final model, persisted with its artifacts
   │
   ├──► optimize_threshold_challenger.py   cost-based decision threshold
   ├──► explain_model.py                   SHAP: global + per-case
   └──► fairness_check_sex.py              ablation test
   │
   ▼
config/model_config.json     single source of truth: active model,
   │                         threshold, cost assumptions
   ▼
src/api (FastAPI)   ◄────────────────────  src/models/*.joblib
   │
   ▼
dashboard (Streamlit)
```

Every stage that fits something to data  the encoder, the scaler,
the imputation medians, SMOTE  is fit exclusively on the training
split and persisted as an artifact, so that a prediction served by the
API is transformed identically to how the model was trained. See the
module-level docstring in each file under `src/` for the specific
contract of that stage.

## Project structure

```
config/model_config.json     active model, threshold, cost assumptions
src/
  data/          preprocess.py
  features/      build_features.py
  models/        train_pipeline.py, train_challenger.py,
                 cross_validate_*.py, optimize_threshold*.py,
                 fairness_check_sex.py
  explainability/ explain_model.py
  api/           FastAPI app (schemas, inference, main)
  utils/         config.py, metrics.py
dashboard/       app.py (Streamlit)
test/            pytest suite (20 tests)
models/artifacts/  trained models, encoders, scalers, figures, reports
FAIRNESS.md       fairness finding and decision log
```

---

## Running it

### 1. Environment

```bash
python -m venv .venv
& .venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

### 2. Rebuild the pipeline from raw data

```bash
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.cross_validate_baseline
python -m src.models.train_pipeline
python -m src.models.train_baseline
python -m src.models.cross_validate_challenger
python -m src.models.train_challenger
python -m src.models.optimize_threshold_challenger
python -m src.explainability.explain_model
python -m src.models.fairness_check_sex
```

Each step reads the previous step's output and writes its own to
`data/` or `models/artifacts/`; none of them mutate shared state, so
the sequence can be re-run in full at any time.

### 3. Serve the model

```bash
uvicorn src.api.main:app --reload --port 8000
```
Interactive docs at `http://127.0.0.1:8000/docs`.

### 4. Dashboard

With the API running, in a second terminal:
```bash
streamlit run dashboard/app.py
```

### 5. Tests

```bash
python -m pytest -v
```

---

## A note on scope

`AGE` remains in the model. Age is treated differently from sex under
most lending fairness frameworks  permitted with restrictions rather
than prohibited outright  but that determination was not made here
with actual legal guidance, only noted as a follow-up in
`FAIRNESS.md`. It should not be read as a closed question.

---

*Built iteratively, resumed after a months-long pause, and re-audited
along the way  which is, if anything, a more faithful account of how
real projects get built than a repository that only shows the final
state.*
