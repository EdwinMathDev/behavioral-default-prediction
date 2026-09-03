"""
train_pipeline_v2.py
======================

Stage 3/3 of the v2 data preparation pipeline — Home Credit Default Risk.

Responsibility
--------------
Builds the final training and test splits for modeling from
data/features/home_credit_features.csv, applying categorical encoding,
skewed-variable transformation, imputation, and scaling.

All stateful transformations (encoder, scaler, imputation medians) are
fitted exclusively on the training split and applied (transform only) to
the test split — same non-negotiable discipline as v1's train_pipeline.py.

Design decisions carried over from v1's lessons
--------------------------------------------------
- No SMOTE. v1 found that SMOTE's linear interpolation over one-hot
  encoded categorical columns produces fractional "synthetic categories"
  that don't correspond to any real value (see v1 train_pipeline.py /
  cross_validate_baseline.py history). Class imbalance here is handled at
  the MODEL level instead (class_weight='balanced' for Logistic
  Regression, scale_pos_weight for XGBoost), decided once a model is
  actually being trained, not baked into the data itself.
- log1p transform uses the sign-aware form (np.sign(x) * np.log1p(abs(x)))
  rather than plain log1p, following the v1 bug where plain log1p produced
  -inf/NaN warnings on columns with legitimate negative values.
- CODE_GENDER is NOT excluded here. Following the same evidence-first
  process used for SEX in v1 (SHAP ablation BEFORE removal, documented in
  FAIRNESS.md), this column stays in the pipeline until an equivalent
  fairness audit is run against a trained model. Excluding it now, without
  evidence, would be no more rigorous than including it blindly.

Pipeline position
------------------
    build_features_v2.py -> [train_pipeline_v2.py] -> modeling

Transformation sequence
------------------------
    1. Load data/features/home_credit_features.csv
    2. Stratified train_test_split on TARGET
    3. Fill missing categoricals with an explicit "Missing" category
       (missingness itself can be informative — e.g. OCCUPATION_TYPE is
       ~31% missing in this dataset, plausibly correlated with employment
       status)
    4. One-hot encode categorical variables (fit on train)
    5. Sign-aware log1p transform of skewed monetary variables
    6. Imputation of remaining missing values (train median)
    7. Standardization of numeric variables (fit on train)

Input
-----
    data/features/home_credit_features.csv

Output
------
    data/features/train_final_v2.csv
    data/features/test_final_v2.csv
    models/artifacts/encoder_v2.joblib
    models/artifacts/scaler_v2.joblib
    models/artifacts/impute_medians_v2.joblib
    models/artifacts/feature_columns_v2.joblib
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TARGET_COL = "TARGET"

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
]

# Monetary/skewed columns worth compressing with a sign-aware log1p.
# BUREAU_TOTAL_DEBT and the *_MAX_OVERDUE_DAYS-type columns can plausibly
# be 0 or, in bureau's case, reflect credit balances — using the safe
# sign-aware transform categorically avoids the v1 log1p bug regardless.
SKEWED_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "BUREAU_TOTAL_DEBT",
]

FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"

ID_COLS = ["SK_ID_CURR"]  # kept out of X, not used as a feature


def load_features(filename: str = "home_credit_features.csv") -> pd.DataFrame:
    path = os.path.join(FEATURES_DIR, filename)
    df = pd.read_csv(path)
    logging.info(f"Features loaded: {df.shape}")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL] + ID_COLS)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Split: train={X_train.shape}, test={X_test.shape}")
    logging.info(f"Class balance train: {y_train.value_counts(normalize=True).to_dict()}")
    logging.info(f"Class balance test:  {y_test.value_counts(normalize=True).to_dict()}")
    return X_train, X_test, y_train, y_test


def fill_categorical_missing(X_train, X_test, cat_cols=CATEGORICAL_COLS):
    present_cols = [c for c in cat_cols if c in X_train.columns]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[present_cols] = X_train[present_cols].fillna("Missing")
    X_test[present_cols] = X_test[present_cols].fillna("Missing")
    logging.info(f"Missing categoricals filled with explicit 'Missing' category: {present_cols}")
    return X_train, X_test


def encode_categoricals(X_train, X_test, cat_cols=CATEGORICAL_COLS):
    present_cols = [c for c in cat_cols if c in X_train.columns]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[present_cols])  # fit ONLY on train

    train_enc = pd.DataFrame(
        encoder.transform(X_train[present_cols]),
        columns=encoder.get_feature_names_out(present_cols),
        index=X_train.index,
    )
    test_enc = pd.DataFrame(
        encoder.transform(X_test[present_cols]),
        columns=encoder.get_feature_names_out(present_cols),
        index=X_test.index,
    )

    X_train = pd.concat([X_train.drop(columns=present_cols), train_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=present_cols), test_enc], axis=1)
    logging.info(f"Encoding applied (fit on train). New columns: {len(train_enc.columns)}")
    return X_train, X_test, encoder


def transform_skewed(X_train, X_test, skewed_cols=SKEWED_COLS):
    present_cols = [c for c in skewed_cols if c in X_train.columns]
    for col in present_cols:
        # Sign-aware log1p — safe even if a column happens to contain
        # negative values, avoiding the v1 log1p -inf/NaN bug.
        X_train[col] = np.sign(X_train[col]) * np.log1p(np.abs(X_train[col]))
        X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))
    logging.info(f"Sign-aware log1p applied to skewed columns: {present_cols}")
    return X_train, X_test


def scale_numeric(X_train, X_test):
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])  # fit ONLY on train
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    logging.info(f"Scaling applied (fit on train) to {len(num_cols)} numeric columns.")
    return X_train, X_test, scaler


def impute_remaining_na(X_train, X_test):
    """Remaining NaNs — the rate/max/avg columns intentionally left as NaN
    by build_features_v2.py for clients with no history in a given table —
    get resolved here with the TRAIN median only, same discipline as v1."""
    medians = X_train.median(numeric_only=True)
    n_train_na = X_train.isna().sum().sum()
    n_test_na = X_test.isna().sum().sum()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    logging.info(f"NaNs imputed with train median — train: {n_train_na}, test: {n_test_na}")
    return X_train, X_test, medians


def run_pipeline():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = load_features()
    X_train, X_test, y_train, y_test = split_data(df)

    X_train, X_test = fill_categorical_missing(X_train, X_test)
    X_train, X_test, encoder = encode_categoricals(X_train, X_test)
    X_train, X_test = transform_skewed(X_train, X_test)
    X_train, X_test, medians = impute_remaining_na(X_train, X_test)
    X_train, X_test, scaler = scale_numeric(X_train, X_test)

    X_train.assign(**{TARGET_COL: y_train.values}).to_csv(
        os.path.join(FEATURES_DIR, "train_final_v2.csv"), index=False
    )
    X_test.assign(**{TARGET_COL: y_test.values}).to_csv(
        os.path.join(FEATURES_DIR, "test_final_v2.csv"), index=False
    )

    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, "encoder_v2.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler_v2.joblib"))
    joblib.dump(medians, os.path.join(ARTIFACTS_DIR, "impute_medians_v2.joblib"))
    joblib.dump(list(X_train.columns), os.path.join(ARTIFACTS_DIR, "feature_columns_v2.joblib"))

    logging.info("v2 modeling-prep pipeline complete.")
    logging.info(f"Artifacts saved in {ARTIFACTS_DIR}/ "
                 f"(encoder_v2.joblib, scaler_v2.joblib, impute_medians_v2.joblib, feature_columns_v2.joblib)")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()
