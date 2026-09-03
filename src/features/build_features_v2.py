"""
build_features_v2.py
=====================

Stage 2/3 of the v2 data preparation pipeline — Home Credit Default Risk.

Responsibility
--------------
Builds a single, client-level feature table (one row per SK_ID_CURR) from
the 5-table Home Credit relational dataset. Each auxiliary table has a
one-to-many relationship with the client and is aggregated down to a
handful of features validated during exploratory analysis
(notebooks/03_eda_bureau.ipynb through 06_eda_credit_card_balance.ipynb) —
not every possible column, only the ones that showed real signal against
the target.

Signal strength ranking found during EDA (strongest to weakest):
    1. CC_AVG_UTILIZATION       (credit_card_balance.csv) — dose-response, +183% relative
    2. PCT_INSTALLMENTS_LATE    (installments_payments.csv) — dose-response, +66% relative
    3. EVER_OVERDUE             (bureau.csv) — binary, ~2x relative
    4. EVER_REFUSED             (previous_application.csv) — binary, +48% relative
    5. HAS_BUREAU_HISTORY       (bureau.csv) — counterintuitive: no history = higher risk
    6. CC_EVER_DPD              (credit_card_balance.csv) — significant but small (p=0.0167)

Missing-value convention
--------------------------
- COUNT-type aggregates (e.g. BUREAU_COUNT) are filled with 0 for clients
  with no records in that table — a client with zero bureau entries
  genuinely has zero credits on record, this is a fact, not a gap.
- RATE/MAX/AVG-type aggregates (e.g. CC_AVG_UTILIZATION) are left as NaN
  for clients with no records in that table. Following the same discipline
  as v1's build_features.py: real imputation happens downstream, fit only
  on the training split, never eagerly here.
- HAS_*_HISTORY flags are always 0/1, never NaN (absence of records IS the
  0 case for these).

Pipeline position
------------------
    raw Home Credit tables -> [build_features_v2.py] -> train_pipeline_v2.py

Input
-----
    data/raw/home_credit/application_train.csv
    data/raw/home_credit/bureau.csv
    data/raw/home_credit/previous_application.csv
    data/raw/home_credit/installments_payments.csv
    data/raw/home_credit/credit_card_balance.csv

Output
------
    data/features/home_credit_features.csv
"""

import os
import gc
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_DIR = "data/raw/home_credit"
FEATURES_DIR = "data/features"

# Columns kept from application_train.csv itself — the heaviest-missing
# housing columns (60-70%+ NaN, see notebooks/02_eda_home_credit.ipynb)
# are deliberately excluded rather than imputed away silently.
APPLICATION_BASE_COLUMNS = [
    "SK_ID_CURR", "TARGET",
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
]

DAYS_EMPLOYED_ANOMALY = 365243


def load_application(train: bool = True) -> pd.DataFrame:
    filename = "application_train.csv" if train else "application_test.csv"
    df = pd.read_csv(os.path.join(RAW_DIR, filename))

    available_cols = [c for c in APPLICATION_BASE_COLUMNS if c in df.columns]
    df = df[available_cols].copy()

    # Age and employment length, in years — same DAYS_EMPLOYED anomaly
    # handling validated in notebooks/02_eda_home_credit.ipynb
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = np.where(
        df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY,
        np.nan,
        -df["DAYS_EMPLOYED"] / 365.25
    )
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])

    logging.info(f"Loaded {filename}: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def aggregate_bureau() -> pd.DataFrame:
    bureau_cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM_DEBT"]
    bureau = pd.read_csv(os.path.join(RAW_DIR, "bureau.csv"), usecols=bureau_cols)

    summary = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        BUREAU_MAX_OVERDUE_DAYS=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
    ).reset_index()

    ever_overdue = bureau.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].apply(
        lambda x: (x > 0).any()
    ).astype(int).rename("EVER_OVERDUE")

    summary = summary.merge(ever_overdue, on="SK_ID_CURR", how="left")
    summary["HAS_BUREAU_HISTORY"] = 1  # every row here has at least 1 bureau record by construction

    del bureau
    gc.collect()
    logging.info(f"Aggregated bureau.csv: {summary.shape[0]:,} clients with bureau history.")
    return summary


def aggregate_previous_application() -> pd.DataFrame:
    prev_cols = ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"]
    prev = pd.read_csv(os.path.join(RAW_DIR, "previous_application.csv"), usecols=prev_cols)

    summary = prev.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        PREV_CANCELED_COUNT=("NAME_CONTRACT_STATUS", lambda x: (x == "Canceled").sum()),
    ).reset_index()

    summary["EVER_REFUSED"] = (summary["PREV_REFUSED_COUNT"] > 0).astype(int)
    summary["EVER_CANCELED"] = (summary["PREV_CANCELED_COUNT"] > 0).astype(int)
    summary["REFUSAL_RATE"] = summary["PREV_REFUSED_COUNT"] / summary["PREV_APP_COUNT"]
    summary["HAS_PREV_APP_HISTORY"] = 1

    del prev
    gc.collect()
    logging.info(f"Aggregated previous_application.csv: {summary.shape[0]:,} clients with prior application history.")
    return summary


def aggregate_installments() -> pd.DataFrame:
    installments_cols = ["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"]
    installments = pd.read_csv(os.path.join(RAW_DIR, "installments_payments.csv"), usecols=installments_cols)

    # Only rows with an actual recorded payment carry a meaningful
    # DAYS_LATE / shortfall — see notebooks/05_eda_installments_payments.ipynb section 3
    paid = installments.dropna(subset=["DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"]).copy()
    paid["DAYS_LATE"] = paid["DAYS_ENTRY_PAYMENT"] - paid["DAYS_INSTALMENT"]
    paid["AMT_SHORTFALL"] = paid["AMT_INSTALMENT"] - paid["AMT_PAYMENT"]

    # Row-level boolean/helper columns computed ONCE, then aggregated with
    # pandas' vectorized groupby().agg() — NOT groupby().apply(lambda g: ...),
    # which processes each of ~339,000 groups through a Python-level
    # function call and is prohibitively memory/time heavy on a 13.6M-row
    # table (this is what caused the ArrayMemoryError).
    paid["IS_LATE"] = paid["DAYS_LATE"] > 0
    paid["IS_UNDERPAID"] = paid["AMT_SHORTFALL"] > 0
    paid["IS_SEVERELY_LATE"] = paid["DAYS_LATE"] >= 30
    paid["DAYS_LATE_IF_LATE"] = np.where(paid["IS_LATE"], paid["DAYS_LATE"], np.nan)

    summary = paid.groupby("SK_ID_CURR").agg(
        PCT_INSTALLMENTS_LATE=("IS_LATE", "mean"),
        AVG_DAYS_LATE=("DAYS_LATE_IF_LATE", "mean"),  # NaN-mean auto-skips non-late rows
        MAX_DAYS_LATE=("DAYS_LATE", "max"),
        EVER_SEVERELY_LATE=("IS_SEVERELY_LATE", "max"),  # max of booleans == "any"
        PCT_INSTALLMENTS_UNDERPAID=("IS_UNDERPAID", "mean"),
    ).reset_index()

    # Clients with zero late installments get NaN from the AVG_DAYS_LATE
    # mean-of-nothing — that IS zero average lateness, not missing data.
    summary["AVG_DAYS_LATE"] = summary["AVG_DAYS_LATE"].fillna(0)
    summary["EVER_SEVERELY_LATE"] = summary["EVER_SEVERELY_LATE"].astype(int)
    summary["HAS_INSTALLMENT_HISTORY"] = 1

    del installments, paid
    gc.collect()
    logging.info(f"Aggregated installments_payments.csv: {summary.shape[0]:,} clients with installment history.")
    return summary


def aggregate_credit_card_balance() -> pd.DataFrame:
    cc_cols = ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD"]
    cc = pd.read_csv(os.path.join(RAW_DIR, "credit_card_balance.csv"), usecols=cc_cols)

    cc["UTILIZATION"] = np.where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
        cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"],
        np.nan
    )
    # Same vectorized groupby().agg() pattern as aggregate_installments() —
    # avoids the memory-heavy groupby().apply(lambda g: pd.Series(...)).
    cc["HAS_DPD"] = cc["SK_DPD"] > 0

    summary = cc.groupby("SK_ID_CURR").agg(
        CC_AVG_UTILIZATION=("UTILIZATION", "mean"),
        CC_MAX_UTILIZATION=("UTILIZATION", "max"),
        CC_PCT_MONTHS_WITH_DPD=("HAS_DPD", "mean"),
        CC_MAX_DPD=("SK_DPD", "max"),
        CC_EVER_DPD=("HAS_DPD", "max"),  # max of booleans == "any"
    ).reset_index()

    summary["CC_EVER_DPD"] = summary["CC_EVER_DPD"].astype(int)
    summary["HAS_CC_HISTORY"] = 1

    del cc
    gc.collect()
    logging.info(f"Aggregated credit_card_balance.csv: {summary.shape[0]:,} clients with credit card history.")
    return summary


# Columns that should become 0 (not NaN) for clients with no record in that
# table, because "0 occurrences among 0 records" is a true fact, not a gap.
COUNT_TYPE_COLUMNS = [
    "BUREAU_COUNT", "BUREAU_ACTIVE_COUNT", "EVER_OVERDUE", "HAS_BUREAU_HISTORY",
    "PREV_APP_COUNT", "PREV_REFUSED_COUNT", "PREV_CANCELED_COUNT",
    "EVER_REFUSED", "EVER_CANCELED", "HAS_PREV_APP_HISTORY",
    "EVER_SEVERELY_LATE", "HAS_INSTALLMENT_HISTORY",
    "CC_EVER_DPD", "HAS_CC_HISTORY",
]


def assemble_features(train: bool = True) -> pd.DataFrame:
    app = load_application(train=train)

    bureau_summary = aggregate_bureau()
    df = app.merge(bureau_summary, on="SK_ID_CURR", how="left")
    del bureau_summary
    gc.collect()

    prev_summary = aggregate_previous_application()
    df = df.merge(prev_summary, on="SK_ID_CURR", how="left")
    del prev_summary
    gc.collect()

    installments_summary = aggregate_installments()
    df = df.merge(installments_summary, on="SK_ID_CURR", how="left")
    del installments_summary
    gc.collect()

    cc_summary = aggregate_credit_card_balance()
    df = df.merge(cc_summary, on="SK_ID_CURR", how="left")
    del cc_summary
    gc.collect()

    # Apply the missing-value convention documented at the top of this file:
    # counts/flags -> 0 for clients absent from that table; rates/max/avg
    # values are deliberately left as NaN for downstream, train-only imputation.
    present_count_cols = [c for c in COUNT_TYPE_COLUMNS if c in df.columns]
    df[present_count_cols] = df[present_count_cols].fillna(0)

    n_nan_after = df.isna().sum().sum()
    logging.info(f"Assembled feature table: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    logging.info(f"Remaining NaNs (rate/max/avg columns, left for train-only imputation downstream): {n_nan_after:,}")

    return df


if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)

    df_features = assemble_features(train=True)
    output_path = os.path.join(FEATURES_DIR, "home_credit_features.csv")
    df_features.to_csv(output_path, index=False)
    logging.info(f"Saved to {output_path}")
