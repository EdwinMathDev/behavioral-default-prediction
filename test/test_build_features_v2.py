"""
test_build_features_v2.py
============================

Unit tests for src.features.build_features_v2 — Home Credit Default Risk (v2).

Uses small synthetic DataFrames injected into each aggregation function
(load_application, aggregate_bureau, aggregate_previous_application,
aggregate_installments, aggregate_credit_card_balance) rather than the
real 5 Kaggle CSVs — these tests run without needing the raw data files
downloaded, and run in milliseconds instead of minutes.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features_v2 import (
    APPLICATION_BASE_COLUMNS,
    load_application,
    aggregate_bureau,
    aggregate_previous_application,
    aggregate_installments,
    aggregate_credit_card_balance,
)


# ---------------------------------------------------------------------
# Fairness: CODE_GENDER must never re-enter the feature set silently
# ---------------------------------------------------------------------

def test_gender_not_in_base_columns():
    assert "CODE_GENDER" not in APPLICATION_BASE_COLUMNS


def test_load_application_excludes_gender():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "CODE_GENDER": ["M", "F"],
        "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans"],
        "AMT_INCOME_TOTAL": [100000, 150000],
        "AMT_CREDIT": [200000, 250000],
        "DAYS_BIRTH": [-10000, -12000],
        "DAYS_EMPLOYED": [-1000, 365243],  # second row is the anomaly placeholder
    })
    result = load_application(train=True, df=synthetic)
    assert "CODE_GENDER" not in result.columns


def test_load_application_handles_days_employed_anomaly():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "DAYS_BIRTH": [-10000, -12000],
        "DAYS_EMPLOYED": [-1000, 365243],
    })
    result = load_application(train=True, df=synthetic)
    assert pd.isna(result.loc[1, "YEARS_EMPLOYED"]), "The 365243 placeholder must become NaN, not a real value"
    assert result.loc[0, "YEARS_EMPLOYED"] == pytest.approx(1000 / 365.25)


# ---------------------------------------------------------------------
# aggregate_bureau
# ---------------------------------------------------------------------

def test_aggregate_bureau_ever_overdue_flag():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
        "CREDIT_DAY_OVERDUE": [0, 15, 0],
        "AMT_CREDIT_SUM_DEBT": [1000, 0, 500],
    })
    result = aggregate_bureau(df=synthetic)

    client_1 = result[result["SK_ID_CURR"] == 1].iloc[0]
    client_2 = result[result["SK_ID_CURR"] == 2].iloc[0]

    assert client_1["EVER_OVERDUE"] == 1, "Client 1 had one overdue record (15 days) — should be flagged"
    assert client_2["EVER_OVERDUE"] == 0, "Client 2 never had an overdue record"
    assert client_1["BUREAU_COUNT"] == 2
    assert client_1["BUREAU_ACTIVE_COUNT"] == 1


def test_aggregate_bureau_has_history_flag_is_always_one():
    # Every row in the RAW bureau table represents a client WITH at least
    # one record — HAS_BUREAU_HISTORY is filled with 0 later, in
    # assemble_features(), only for clients missing from this table entirely.
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_BUREAU": [10],
        "CREDIT_ACTIVE": ["Active"],
        "CREDIT_DAY_OVERDUE": [0],
        "AMT_CREDIT_SUM_DEBT": [0],
    })
    result = aggregate_bureau(df=synthetic)
    assert (result["HAS_BUREAU_HISTORY"] == 1).all()


# ---------------------------------------------------------------------
# aggregate_previous_application
# ---------------------------------------------------------------------

def test_aggregate_previous_application_ever_refused():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [100, 101, 102],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
    })
    result = aggregate_previous_application(df=synthetic)

    client_1 = result[result["SK_ID_CURR"] == 1].iloc[0]
    client_2 = result[result["SK_ID_CURR"] == 2].iloc[0]

    assert client_1["EVER_REFUSED"] == 1
    assert client_2["EVER_REFUSED"] == 0
    assert client_1["REFUSAL_RATE"] == pytest.approx(0.5)
    assert client_2["REFUSAL_RATE"] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# aggregate_installments
# ---------------------------------------------------------------------

def test_aggregate_installments_lateness_and_shortfall():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "DAYS_INSTALMENT": [-100, -70, -50],
        "DAYS_ENTRY_PAYMENT": [-100, -50, -50],  # client 1: on time, then 20 days late
        "AMT_INSTALMENT": [1000, 1000, 500],
        "AMT_PAYMENT": [1000, 800, 500],  # client 1's second payment is short by 200
    })
    result = aggregate_installments(df=synthetic)

    client_1 = result[result["SK_ID_CURR"] == 1].iloc[0]
    client_2 = result[result["SK_ID_CURR"] == 2].iloc[0]

    assert client_1["PCT_INSTALLMENTS_LATE"] == pytest.approx(0.5), "1 of 2 installments was late"
    assert client_1["MAX_DAYS_LATE"] == 20
    assert client_1["PCT_INSTALLMENTS_UNDERPAID"] == pytest.approx(0.5)
    assert client_2["PCT_INSTALLMENTS_LATE"] == pytest.approx(0.0), "Client 2 paid exactly on time"
    assert client_2["AVG_DAYS_LATE"] == 0, "No late payments -> average lateness must be 0, not NaN"


def test_aggregate_installments_drops_unpaid_rows():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "DAYS_INSTALMENT": [-100, -70],
        "DAYS_ENTRY_PAYMENT": [-100, np.nan],  # second installment was never paid
        "AMT_INSTALMENT": [1000, 1000],
        "AMT_PAYMENT": [1000, np.nan],
    })
    result = aggregate_installments(df=synthetic)
    client_1 = result[result["SK_ID_CURR"] == 1].iloc[0]
    # Only the 1 genuinely-paid installment should count -- the unpaid one is dropped.
    assert client_1["PCT_INSTALLMENTS_LATE"] == pytest.approx(0.0)


# ---------------------------------------------------------------------
# aggregate_credit_card_balance
# ---------------------------------------------------------------------

def test_aggregate_credit_card_balance_utilization_and_dpd():
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "AMT_BALANCE": [5000, 8000, 0],
        "AMT_CREDIT_LIMIT_ACTUAL": [10000, 10000, 5000],
        "SK_DPD": [0, 10, 0],
    })
    result = aggregate_credit_card_balance(df=synthetic)

    client_1 = result[result["SK_ID_CURR"] == 1].iloc[0]
    client_2 = result[result["SK_ID_CURR"] == 2].iloc[0]

    assert client_1["CC_AVG_UTILIZATION"] == pytest.approx(0.65)  # mean of 0.5 and 0.8
    assert client_1["CC_EVER_DPD"] == 1
    assert client_2["CC_EVER_DPD"] == 0
    assert client_2["CC_AVG_UTILIZATION"] == pytest.approx(0.0)


def test_aggregate_credit_card_balance_zero_limit_is_nan_not_error():
    """A credit limit of 0 must not raise a ZeroDivisionError or produce
    an infinite utilization ratio — it should become NaN, resolved by
    downstream imputation, same discipline as v1's ratio handling."""
    synthetic = pd.DataFrame({
        "SK_ID_CURR": [1],
        "AMT_BALANCE": [500],
        "AMT_CREDIT_LIMIT_ACTUAL": [0],
        "SK_DPD": [0],
    })
    result = aggregate_credit_card_balance(df=synthetic)
    assert not np.isinf(result["CC_AVG_UTILIZATION"]).any()
    assert not np.isinf(result["CC_MAX_UTILIZATION"]).any()
