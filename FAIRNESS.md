# Fairness Decision Log — Credit Risk Engine

## SEX excluded from the model (2026-08-14)

### Finding

SHAP analysis of the XGBoost model (`src/explainability/explain_model.py`)
showed `SEX` (one-hot encoded as `SEX_1.0` / `SEX_2.0`) ranking 6th of 19
features by mean absolute SHAP value, with a clear, systematic directional
effect on the predicted default probability — not a marginal or noisy
contribution.

Using sex as a direct input to a credit decision is prohibited or heavily
restricted under most consumer-lending anti-discrimination regulations
(e.g. the Equal Credit Opportunity Act in the US, and equivalent laws
elsewhere), regardless of whether it is statistically predictive.

### Test performed

An ablation test (`src/models/fairness_check_sex.py`) trained two otherwise
identical XGBoost models on the same train/test split — one with `SEX`
included, one without — and compared performance:

| Metric | With SEX | Without SEX | Delta |
|---|---|---|---|
| AUC-ROC | 0.7711 | 0.7699 | +0.0011 |
| KS statistic | 0.4066 | 0.4089 | -0.0023 (better without) |
| Recall | 0.4808 | 0.4785 | +0.0023 |
| False Negatives | 689 | 692 | -3 |

### Decision

**`SEX` is permanently removed from the feature set**, at every stage of
the pipeline (`preprocess.py`, `build_features.py`, `train_pipeline.py`,
`cross_validate_baseline.py`). The AUC difference (0.0011) is noise —
roughly 15x smaller than the margin (0.019) that was treated as a
meaningful gain when XGBoost was promoted over the Logistic Regression
baseline. There is no predictive-performance justification for retaining
a protected attribute in a lending model.

### Follow-up

- All models trained before this date (`logreg_baseline.joblib`,
  the original `xgb_challenger.joblib`) were trained WITH `SEX` and
  should be considered superseded once the pipeline is re-run.
- `AGE` was intentionally kept for now — many jurisdictions permit its
  use in credit scoring with restrictions (e.g. no discrimination against
  elderly applicants), unlike sex, which is a harder line in most
  frameworks. This should be revisited with actual legal/compliance
  guidance before production deployment, not assumed.
- Any future feature added to this project should be checked against
  SHAP feature importance before being treated as production-ready,
  not just for predictive value but for fairness risk.
