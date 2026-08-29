# Behavioral Default Prediction

A credit-default scoring system built around a simple premise: a model
is not finished when it achieves a good AUC. It is finished when its
predictions are reproducible, its decisions are explainable, its
threshold reflects the economics of the business it serves, and every
promotion decision — including "we should use the more complex
model" — is backed by a statistical test, not by which number looked
bigger.

This repository documents that process end to end, including the
points where an earlier version of the pipeline was wrong, and the
point where a previously-documented decision was formally reversed
after being re-examined with more rigor.

---

## What this is

**v1** of a model that predicts the probability that a credit-card
holder will default on payment in the following month, built on the
UCI *Default of Credit Card Clients* dataset (Taiwan) — inherited from
an earlier project, [`credit-risk-engine`](https://github.com/EdwinMathDev),
and re-purposed here as the foundation for a genuinely *behavioral*
default-prediction system.

**v2** (in progress, see [Roadmap](#roadmap)) will extend this to the
Home Credit Default Risk dataset, which has real multi-table
behavioral history (bureau records, prior loans, installment payment
behavior) — the kind of data this project's name actually promises.

| Metric / Feature | Value / Details |
|---|---|
| **Active model** | Logistic Regression |
| **AUC-ROC** | 0.752 (single split) · 0.766 ± 0.006 (5-fold CV) |
| **KS statistic** | 0.397 (single split) · 0.413 ± 0.009 (5-fold CV) |
| **Decision threshold** | 0.410 (cost-optimal, chosen from out-of-fold predictions — never from the test set) |
| **Fairness** | `SEX` excluded from every stage of the pipeline — see [`FAIRNESS.md`](FAIRNESS.md) |

## Why Logistic Regression, and not XGBoost

An XGBoost challenger was originally promoted over the Logistic
Regression baseline (documented history in
`config/model_config.json → model.promotion_notes`), on the basis of a
5-fold CV showing "improvement in 4 of 5 folds." That is a real
observation, but it is not the same thing as a statistically
significant difference.

Because both models were cross-validated on the *exact same* 5 folds
(`StratifiedKFold(random_state=42)`, shared between
`cross_validate_baseline.py` and `cross_validate_challenger.py`), the
comparison could be — and was — redone properly as a **paired t-test**
over the per-fold differences (`compare_baseline_vs_challenger.py`):

| Metric | Baseline (LogReg) | Challenger (XGBoost) | Diff | p-value | Significant at 5%? |
|---|---|---|---|---|---|
| AUC-ROC | 0.7664 | 0.7727 | +0.0062 | 0.0609 | No |
| KS statistic | 0.4127 | 0.4173 | +0.0046 | 0.3237 | No |

Neither difference clears the conventional significance threshold —
and in one of the five folds, XGBoost actually performed *worse* than
the baseline, which a simple average of "4 of 5 folds" obscures. With
no statistically defensible performance gain, the added complexity
and loss of interpretability of XGBoost isn't justified. **The
promotion was reversed**, and the reasoning for the reversal is
recorded in full in `model_config.json`, right next to the original
promotion notes it replaces — nothing was quietly deleted.

## Choosing the decision threshold without cheating

The original threshold-optimization script chose its cutoff *and*
reported final metrics on the same test set — a subtle leakage that
optimistically biases the reported numbers toward that one split.
This was fixed (`generate_oof_predictions.py` +
`optimize_threshold_v2.py`): the threshold is selected using
out-of-fold predictions (each row scored by a model that never saw it
in training), and `test_final.csv` is touched exactly once, only to
report how the already-decided threshold performs.

The threshold itself (0.410) reflects an explicit, documented cost
assumption — missing a real default is assumed to cost 5× more than
wrongly flagging a good client (`business_cost_assumptions` in
`model_config.json`) — not the arbitrary default of 0.5.

## A finding worth stating plainly

`SEX` was excluded from the feature set upstream, in
`build_features.py`, following the fairness ablation documented in
[`FAIRNESS.md`](FAIRNESS.md): a SHAP-based finding that the variable
had a systematic, non-negligible effect on predicted risk with no
performance justification for keeping it. Because it never enters
`build_features.py`'s `BASE_COLUMNS`, it cannot silently re-appear
downstream — anyone attempting to re-add it as a feature would need to
touch that file directly, and would need to justify the fairness
implications explicitly at that point, not have them buried in a
config flag elsewhere.

## A note on scope

`AGE` remains in the model. Age is treated differently from sex under
most lending fairness frameworks — permitted with restrictions rather
than prohibited outright — but that determination was not made here
with actual legal guidance, only noted as a follow-up in
`FAIRNESS.md`. It should not be read as a closed question.

---

## Architecture

```
raw data
   │
   ▼
preprocess.py                cleaning: imputation, category correction
   │
   ▼
build_features.py            domain features: utilization, payment ratios,
   │                         delinquency history, trend, volatility
   │                         (SEX excluded here — see FAIRNESS.md)
   ▼
train_pipeline.py            stratified split → encode → scale → SMOTE
   │                         (all fit on train only — no leakage)
   ▼
cross_validate_baseline.py   5-fold CV, stability check
cross_validate_challenger.py 5-fold CV, same folds as baseline
   │
   ▼
compare_baseline_vs_challenger.py    paired t-test — decides which model wins
   │
   ▼
train_baseline.py            final model, persisted with its artifacts
   │
   ├──► generate_oof_predictions.py   out-of-fold probabilities (no leakage)
   ├──► optimize_threshold_v2.py      cost-based threshold, chosen from OOF only
   ├──► explain_model.py              SHAP (auto-detects Linear vs Tree explainer)
   └──► fairness_check_sex.py         ablation test (legacy — SEX already excluded upstream)
   │
   ▼
config/model_config.json     single source of truth: active model,
   │                         threshold, cost assumptions, promotion history
   ▼
src/api (FastAPI)   ◄────────────────────  models/artifacts/*.joblib
   │
   ▼
dashboard (Streamlit)        reads the same config — never hardcodes
                              which model is "active"
```

Every stage that fits something to data — the encoder, the scaler,
the imputation medians, SMOTE — is fit exclusively on the training
split and persisted as an artifact, so that a prediction served by the
API is transformed identically to how the model was trained. See the
module-level docstring in each file under `src/` for the specific
contract of that stage.

## Project structure

```
config/model_config.json     active model, threshold, cost assumptions, promotion history
src/
  data/           preprocess.py
  features/       build_features.py
  models/         train_pipeline.py, train_baseline.py, train_challenger.py,
                  cross_validate_baseline.py, cross_validate_challenger.py,
                  compare_baseline_vs_challenger.py, generate_oof_predictions.py,
                  optimize_threshold.py, optimize_threshold_v2.py,
                  optimize_threshold_challenger.py, fairness_check_sex.py
  explainability/ explain_model.py
  api/            FastAPI app (schemas, inference, main)
  utils/          config.py, metrics.py
dashboard/        app.py (Streamlit)
test/             pytest suite
models/artifacts/ trained models, encoders, scalers, figures, reports
FAIRNESS.md       fairness finding and decision log
```

---

## Running it

### 1. Environment

```bash
python -m venv .venv
& .venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
pip install -e .
```

> **Important:** always run project scripts as modules
> (`python -m src.models.train_baseline`), never by clicking a "Run"
> button or invoking the file by direct path. This project's editable
> install has, more than once, resolved imports incorrectly when run
> that way — `-m` from the project root is the pattern that reliably
> works.

### 2. Rebuild the pipeline from raw data

```bash
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train_pipeline
python -m src.models.cross_validate_baseline
python -m src.models.cross_validate_challenger
python -m src.models.compare_baseline_vs_challenger
python -m src.models.train_baseline
python -m src.models.generate_oof_predictions
python -m src.models.optimize_threshold_v2
python -m src.explainability.explain_model
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

## Roadmap

- [x] **v1** — inherit and re-validate the `credit-risk-engine` pipeline
      end to end: fix inherited bugs (negative-value `log1p`, seaborn
      deprecations, a stale editable install, a model artifact out of
      sync with its own schema), re-decide the active model with a
      proper statistical test, fix a threshold-selection leakage, and
      ship a working API + dashboard.
- [ ] **v2** — migrate to the Home Credit Default Risk dataset
      (bureau history, prior loans, installment-level payment
      behavior across multiple products), which is where this
      project's name — *behavioral* — actually gets earned. This work
      happens on a separate branch so `main` always reflects a working
      v1.

---

*Built on a pipeline that had been paused for months, resumed and
re-audited stage by stage — which is, if anything, a more faithful
account of how real projects get built than a repository that only
shows the final state.*
