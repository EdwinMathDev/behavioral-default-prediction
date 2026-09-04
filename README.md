# Behavioral Default Prediction

A credit-default scoring system built around a simple premise: a model
is not finished when it achieves a good AUC. It is finished when its
predictions are reproducible, its decisions are explainable, its
threshold reflects the economics of the business it serves, and every
promotion decision — including "we should use the more complex
model" — is backed by a statistical test, not by which number looked
bigger.

This repository documents that process end to end, including the
points where an earlier version of the pipeline was wrong, the point
where a previously-documented decision was formally reversed after
being re-examined with more rigor, and the point where a second
model's apparent edge turned out to depend mostly on data this project
didn't actually build.

---

## What this is

**v1** — a model that predicts the probability that a credit-card
holder will default on payment in the following month, built on the
UCI *Default of Credit Card Clients* dataset (Taiwan) — inherited from
an earlier project, [`credit-risk-engine`](https://github.com/EdwinMathDev),
and re-purposed here as the foundation for a genuinely *behavioral*
default-prediction system. **In production on `main`.**

**v2** — migrated to the Home Credit Default Risk dataset, which has
real multi-table behavioral history (bureau records, prior loans,
installment payment behavior, credit card balances) — the kind of data
this project's name actually promises. Modeling is complete on the
`v2-home-credit` branch (see [Roadmap](#roadmap) for the full,
including-the-uncomfortable-parts writeup); a serving layer (API +
dashboard) has not been built yet.

| Metric / Feature | v1 (Taiwan, `main`) | v2 (Home Credit, `v2-home-credit`) |
|---|---|---|
| **Active model** | Logistic Regression | XGBoost |
| **AUC-ROC** | 0.752 (single split) · 0.766 ± 0.006 (5-fold CV) | 0.765 (single split) · 0.764 ± 0.004 (5-fold CV) |
| **KS statistic** | 0.397 (single split) · 0.413 ± 0.009 (5-fold CV) | 0.397 (single split) · 0.397 ± 0.007 (5-fold CV) |
| **Decision threshold** | 0.410 (cost-optimal, from OOF predictions) | 0.700 (cost-optimal, from OOF predictions) |
| **Fairness** | `SEX` excluded — see [`FAIRNESS.md`](FAIRNESS.md) | `CODE_GENDER` excluded — same method, see Roadmap |
| **Honesty caveat** | XGBoost promotion reversed — no significant edge over Logistic Regression (see below) | Most of the AUC edge over v1 traces to precomputed external scores (`EXT_SOURCE_*`), not this project's own feature engineering — see [Roadmap](#roadmap) |

---

## Why Logistic Regression, and not XGBoost (v1)

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

## Choosing the decision threshold without cheating (v1)

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
`model_config.json`) — not the arbitrary default of 0.5. v2 uses the
same OOF-based method and the same cost ratio (`optimize_threshold_v2_active.py`),
arriving at a different threshold (0.700) because Home Credit's more
extreme class imbalance shifts where that cost minimum sits.

## A finding worth stating plainly (v1)

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

`AGE` remains in the model (both v1 and v2). Age is treated differently
from sex/gender under most lending fairness frameworks — permitted
with restrictions rather than prohibited outright — but that
determination was not made here with actual legal guidance, only noted
as a follow-up in `FAIRNESS.md`. It should not be read as a closed
question.

---

## Architecture — v1 (Taiwan, `main`)

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

## Architecture — v2 (Home Credit, `v2-home-credit`)

```
5 raw Home Credit tables (data/raw/home_credit/)
   application_train.csv, bureau.csv, previous_application.csv,
   installments_payments.csv, credit_card_balance.csv
   │
   ▼  (explored one table at a time — notebooks/02 through 06)
build_features_v2.py         aggregates each auxiliary table down to
   │                         client-level features validated during EDA:
   │                         HAS_*_HISTORY flags, EVER_OVERDUE, EVER_REFUSED,
   │                         PCT_INSTALLMENTS_LATE, CC_AVG_UTILIZATION, etc.
   │                         (CODE_GENDER excluded here — see below)
   ▼
train_pipeline_v2.py         stratified split → encode → scale
   │                         (no SMOTE — class_weight/scale_pos_weight
   │                         instead, a lesson carried over from v1)
   ▼
cross_validate_baseline_v2.py     5-fold CV, Logistic Regression
cross_validate_challenger_v2.py   5-fold CV, XGBoost, same folds
   │
   ▼
compare_baseline_vs_challenger_v2.py   paired t-test — XGBoost wins here
   │                                   (unlike v1), p=0.0002 (AUC), p=0.0005 (KS)
   ▼
train_baseline_v2.py / train_challenger_v2.py   final models, persisted
   │
   ├──► fairness_check_gender_v2.py         CODE_GENDER ablation (+0.0018 AUC → excluded)
   ├──► generate_oof_predictions_v2_xgb.py  out-of-fold probabilities (no leakage)
   ├──► optimize_threshold_v2_active.py     cost-based threshold, reads the ACTIVE
   │                                        model from config — never hardcoded
   ├──► explain_model_v2.py                 SHAP (auto-detects Linear vs Tree explainer)
   ├──► ablation_check_ext_source_v2.py     EXT_SOURCE_1/2/3 ablation — the honesty check
   └──► check_ext_source_missingness.py     confirms no production-availability risk
   │
   ▼
config/model_config_v2.json   single source of truth: active model,
                               threshold, cost assumptions, full promotion
                               history including the EXT_SOURCE finding

(No API/dashboard yet for v2 — see Roadmap.)
```

Same discipline as v1 in both pipelines: every stage that fits
something to data — the encoder, the scaler, the imputation medians —
is fit exclusively on the training split. See the module-level
docstring in each file under `src/` for the specific contract of that
stage.

## Project structure

```
config/
  model_config.json           v1: active model, threshold, cost assumptions, promotion history
  model_config_v2.json        v2: same, for the Home Credit pipeline
src/
  data/           preprocess.py                          (v1)
  features/       build_features.py                       (v1)
                  build_features_v2.py                     (v2)
  models/         train_pipeline.py, train_baseline.py, train_challenger.py,
                  cross_validate_baseline.py, cross_validate_challenger.py,
                  compare_baseline_vs_challenger.py, generate_oof_predictions.py,
                  optimize_threshold.py, optimize_threshold_v2.py,
                  optimize_threshold_challenger.py, fairness_check_sex.py    (v1)
                  train_pipeline_v2.py, train_baseline_v2.py, train_challenger_v2.py,
                  cross_validate_baseline_v2.py, cross_validate_challenger_v2.py,
                  compare_baseline_vs_challenger_v2.py,
                  generate_oof_predictions_v2.py, generate_oof_predictions_v2_xgb.py,
                  optimize_threshold_home_credit.py, optimize_threshold_v2_active.py,
                  fairness_check_gender_v2.py, ablation_check_ext_source_v2.py    (v2)
  explainability/ explain_model.py                        (v1)
                  explain_model_v2.py                       (v2)
  api/            FastAPI app (schemas, inference, main)   (v1 only, so far)
  utils/          config.py, metrics.py                    (shared)
dashboard/        app.py (Streamlit)                       (v1 only, so far)
notebooks/        01_eda.ipynb                              (v1)
                  02_eda_home_credit.ipynb through
                  06_eda_credit_card_balance.ipynb           (v2, one per table)
test/             pytest suite                              (v1)
data/raw/home_credit/   5 raw Home Credit CSVs (gitignored — not versioned)
models/artifacts/ trained models, encoders, scalers, figures, reports (both versions)
FAIRNESS.md       fairness finding and decision log (v1; v2's CODE_GENDER
                  finding lives in model_config_v2.json for now)
check_ext_source_missingness.py   standalone v2 sanity check (project root)
```

---

## Running it — v1

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

## Running it — v2

> Checked out on the `v2-home-credit` branch. Requires the 5 raw Home
> Credit CSVs in `data/raw/home_credit/` (downloaded separately from
> Kaggle — see the competition's data page; not included in this repo).

```bash
python -m src.features.build_features_v2
python -m src.models.train_pipeline_v2
python -m src.models.cross_validate_baseline_v2
python -m src.models.train_baseline_v2
python -m src.models.train_challenger_v2
python -m src.models.cross_validate_challenger_v2
python -m src.models.compare_baseline_vs_challenger_v2
python -m src.models.fairness_check_gender_v2
python -m src.models.generate_oof_predictions_v2_xgb
python -m src.models.optimize_threshold_v2_active
python -m src.explainability.explain_model_v2
python -m src.models.ablation_check_ext_source_v2
python check_ext_source_missingness.py
```

Each step reads the previous step's output and writes its own to
`data/` or `models/artifacts/`; none of them mutate shared state, so
the sequence can be re-run in full at any time. No API/dashboard yet —
see [Roadmap](#roadmap).

---

## Roadmap

- [x] **v1** — inherit and re-validate the `credit-risk-engine` pipeline
      end to end: fix inherited bugs (negative-value `log1p`, seaborn
      deprecations, a stale editable install, a model artifact out of
      sync with its own schema), re-decide the active model with a
      proper statistical test, fix a threshold-selection leakage, and
      ship a working API + dashboard.
- [x] **v2 (exploration + modeling complete)** — migrated to the Home
      Credit Default Risk dataset (bureau history, prior loans,
      installment-level payment behavior, credit card balances across
      5 relational tables). Work happened on a separate `v2-home-credit`
      branch so `main` always reflects a working v1.

      **v2 findings, in brief:**
      - EDA across all 5 tables found genuine behavioral signal with
        clear dose-response patterns: `PCT_INSTALLMENTS_LATE` (+66%
        relative default rate from lowest to highest tercile) and
        `CC_AVG_UTILIZATION` (+183% relative, the strongest signal found
        purely from this project's own engineered features).
      - A Logistic Regression baseline (v2) scored lower than v1's
        (AUC 0.7551 vs. 0.7664, 5-fold CV) despite the richer data —
        traced to Home Credit's more extreme class imbalance (8% vs.
        22% default rate in v1's dataset), not to wasted feature signal
        (confirmed via coefficient inspection: the `HAS_*_HISTORY`
        flags do carry real, appropriately-signed weight).
      - An XGBoost challenger *did* show a statistically significant
        improvement over the v2 baseline this time (unlike v1, where the
        same test led to rejecting XGBoost) — paired t-test AUC +0.0084
        (p=0.0002), KS +0.0144 (p=0.0005), winning in 5/5 folds. Promoted
        to v2 production per `config/model_config_v2.json`.
      - `CODE_GENDER` was audited the same way `SEX` was in v1
        (evidence-first, not excluded by default) and removed after an
        ablation showed negligible performance cost (+0.0018 AUC,
        confirmed via a full 5-fold CV re-run after removal).
      - **Honesty check that matters most**: SHAP showed `EXT_SOURCE_1/2/3`
        — precomputed external scores present in the raw data, not
        features this project engineered — dominating feature
        importance by a wide margin. An ablation confirmed they alone
        are worth +0.0507 AUC; without them, v2 (0.7144) falls clearly
        *below* v1 (0.7664, -0.0520). A follow-up availability check
        found no production-risk case for removing them anyway
        (`EXT_SOURCE_2` missing only 0.2% of the time; only 0.06% of
        clients lack all three, with no meaningful default-rate
        difference), so they remain in the production model — but the
        honest accounting is that the "v2 nearly closes the gap with
        v1" headline number depends mostly on precomputed external
        scores, not on this project's own multi-table behavioral
        feature engineering. That engineering is real and individually
        validated (see the dose-response findings above), but it is a
        secondary contributor to the model's overall performance, not
        the primary one. Full accounting in
        `config/model_config_v2.json`'s `promotion_notes`.
- [ ] **v2 (deployment parity)** — API + dashboard for the v2 model,
      matching v1's. Not yet built; v2 currently exists as a validated
      modeling pipeline (`src/features/build_features_v2.py` through
      `src/explainability/explain_model_v2.py`) without a serving layer.

---

*Built on a pipeline that had been paused for months, resumed and
re-audited stage by stage — which is, if anything, a more faithful
account of how real projects get built than a repository that only
shows the final state.*
