"""
build_features.py
==================

Stage 2/3 of the data preparation pipeline — Credit Risk Engine.

Responsibility
--------------
Builds domain-specific features from the six-month billing and
payment history: credit utilization ratios, payment ratios,
month-over-month trends, billing/payment volatility, delinquency
history, and overall payment capacity.

These features are always computed on cleaned but UNSCALED data
(see preprocess.py), so that they retain a direct financial
interpretation. Encoding, scaling, and class balancing are applied
in a later stage (train_pipeline.py) to avoid information leakage
between the train and test splits.

Pipeline position
------------------
    raw data -> preprocess.py -> [build_features.py] -> train_pipeline.py

Input
-----
    data/processed/credit_card_default_clean.csv

Output
------
    data/features/credit_card_features.csv
"""

import logging
import pandas as pd
import numpy as np
from src.data.load_data import load_raw_data, save_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_COLUMNS = [
    "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default payment_next_month"
]
# SEX removida — ver FAIRNESS.md para el detalle del hallazgo y la decisión.

PAY_DELAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Tope razonable para ratios que en teoría podrían dispararse
# (ej. alguien paga mucho más de lo que facturó ese mes)
RATIO_CLIP_MAX = 3.0


def create_features(df: pd.DataFrame) -> pd.DataFrame:

    # ── 1. Filtrar columnas base ──────────────────────────────
    logging.info("Filtrando columnas relevantes...")
    df = df[BASE_COLUMNS].copy()

    # ── 2. Ratios de utilización y pago por mes ───────────────
    logging.info("Creando features de utilización y pago por mes...")
    for i in range(1, 7):
        # % del límite usado (clip para cubrir sobregiro moderado)
        df[f"utilization_ratio_{i}"] = (
            df[f"BILL_AMT{i}"] / df["LIMIT_BAL"]
        ).clip(0, 2)

        # % de la deuda pagado ese mes — clip para evitar que pagos
        # desproporcionados (deuda casi 0, pago grande) disparen el ratio
        df[f"payment_ratio_{i}"] = np.where(
            df[f"BILL_AMT{i}"] > 0,
            (df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"]).clip(0, RATIO_CLIP_MAX),
            0
        )

    # ── 3. Tendencias mes a mes ───────────────────────────────
    logging.info("Creando features de tendencia mes a mes...")
    for i in range(1, 6):
        df[f"bill_trend_{i}"] = df[f"BILL_AMT{i+1}"] - df[f"BILL_AMT{i}"]
        df[f"pay_trend_{i}"]  = df[f"PAY_AMT{i+1}"]  - df[f"PAY_AMT{i}"]

    # ── 4. Volatilidad de pagos y facturas ────────────────────
    logging.info("Creando features de volatilidad...")
    df["payment_volatility"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].std(axis=1)
    df["bill_volatility"]    = df[[f"BILL_AMT{i}" for i in range(1, 7)]].std(axis=1)

    # ── 5. Historial de delays ────────────────────────────────
    logging.info("Creando features de historial de delays...")
    df["recent_delay"]   = (df["PAY_0"] > 0).astype(int)
    df["avg_pay_delay"]  = df[PAY_DELAY_COLS].mean(axis=1)
    df["max_pay_delay"]  = df[PAY_DELAY_COLS].max(axis=1)
    df["months_delayed"] = (df[PAY_DELAY_COLS] > 0).sum(axis=1)

    # ── 6. Capacidad de pago global ───────────────────────────
    logging.info("Creando features de capacidad de pago global...")
    df["total_bill"]    = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["total_payment"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["global_payment_ratio"] = np.where(
        df["total_bill"] > 0,
        (df["total_payment"] / df["total_bill"]).clip(0, RATIO_CLIP_MAX),
        0
    )

    # ── 7. Grupo etario ordinal ───────────────────────────────
    logging.info("Creando feature de grupo etario...")
    df["age_group"] = pd.cut(
        df["AGE"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=[0, 1, 2, 3, 4],
        right=False
    ).astype("Int64")

    # ── 8. Sanitizar inf/nan generados por las divisiones ─────
    n_before = df.isna().sum().sum()
    df = df.replace([np.inf, -np.inf], np.nan)
    n_after = df.isna().sum().sum()
    if n_after > n_before:
        logging.warning(f"Se generaron {n_after - n_before} valores infinitos convertidos a NaN.")
    if n_after > 0:
        logging.warning(f"El dataset de features tiene {n_after} NaNs totales — "
                         f"se resolverán en el paso de train/test split (imputación fit-solo-en-train).")

    logging.info(f"Features creadas: {df.shape[1]} columnas — {df.shape[0]} filas.")
    return df


if __name__ == "__main__":
    clean_file = "credit_card_default_clean.csv"
    logging.info(f"Cargando dataset limpio desde data/processed/{clean_file}...")
    df = load_raw_data(clean_file, data_dir="data/processed")

    df_features = create_features(df)

    save_processed_data(df_features, "credit_card_features.csv", data_dir="data/features")
    logging.info("Pipeline de feature engineering completado.")
