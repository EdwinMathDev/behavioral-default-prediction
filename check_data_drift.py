"""
check_data_drift.py
======================

Data and prediction drift monitoring — behavioral-default-prediction.

Responsibility
--------------
Nothing in this project, until now, addresses what happens after a model
ships: real applicant populations shift over time (economic conditions,
underwriting policy changes, a new marketing channel bringing in a
different kind of applicant), and a model trained on last year's
population can quietly degrade without ever throwing an error.

This script implements the Population Stability Index (PSI) — the
standard credit-risk-industry technique for exactly this problem, not a
generic statistical test — comparing a REFERENCE distribution (typically
the training set) against a CURRENT distribution (typically a recent batch
of scored applicants) for:

    1. Each individual feature (has the applicant population's underlying
       characteristics shifted?)
    2. The model's predicted probability itself (has the model's OUTPUT
       distribution shifted, even if you haven't examined every feature?)

Standard PSI interpretation thresholds (used as-is here, not invented for
this project):
    PSI < 0.10             -> no significant shift
    0.10 <= PSI < 0.25      -> moderate shift, worth investigating
    PSI >= 0.25             -> significant shift, the model should be
                               re-validated (and likely retrained) before
                               being trusted further

Usage
-----
    python check_data_drift.py --version v1 \\
        --reference data/features/train_final.csv \\
        --current data/features/test_final.csv

    python check_data_drift.py --version v2 \\
        --reference data/features/train_final_v2.csv \\
        --current data/features/test_final_v2.csv

In real use, "current" would be a recent batch of newly-scored applicants,
not the held-out test set — the test set is used here only to demonstrate
the mechanism with data already on hand, and should show LOW drift (it's
a random split of the same population, not genuinely new data). A
meaningfully non-zero PSI here would itself be worth investigating.

Output
------
    A printed report, plus a JSON file with full per-feature PSI values,
    written next to wherever --current lives.
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25
N_BINS = 10


def calculate_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    PSI = sum( (current_pct - reference_pct) * ln(current_pct / reference_pct) )
    over n_bins, where bin edges are defined by the REFERENCE distribution's
    quantiles — current is measured against bins that made sense for the
    population the model was actually trained on.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # Quantile-based bins from the reference distribution. Constant/near-constant
    # columns can collapse bin edges — duplicates="drop" handles that gracefully.
    try:
        bin_edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    except Exception:
        return np.nan

    if len(bin_edges) < 3:
        # Not enough distinct values to form meaningful bins (e.g. a
        # near-constant or binary column) -- PSI isn't a meaningful
        # measure here; the caller should rely on a simple mean/rate
        # comparison for such columns instead.
        return np.nan

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / max(len(reference), 1)
    cur_pct = cur_counts / max(len(current), 1)

    # Avoid log(0) / division by zero for empty bins -- a tiny epsilon is
    # standard practice for PSI, since a truly empty bin still represents
    # a real (large) shift that shouldn't be silently ignored.
    epsilon = 1e-4
    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def classify_psi(psi: float) -> str:
    if np.isnan(psi):
        return "not applicable"
    if psi < PSI_MODERATE_THRESHOLD:
        return "no significant shift"
    if psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate shift -- investigate"
    return "significant shift -- re-validate model"


def run(reference_path: str, current_path: str, target_col: str, model_path: str = None):
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    numeric_cols = [
        c for c in reference.columns
        if c != target_col and pd.api.types.is_numeric_dtype(reference[c])
    ]

    logging.info(f"Comparing {len(numeric_cols)} numeric features: "
                 f"{os.path.basename(reference_path)} (reference, n={len(reference):,}) "
                 f"vs. {os.path.basename(current_path)} (current, n={len(current):,})")

    feature_results = {}
    for col in numeric_cols:
        psi = calculate_psi(reference[col].values, current[col].values)
        feature_results[col] = {"psi": psi, "status": classify_psi(psi)}

    flagged = {k: v for k, v in feature_results.items() if v["status"] != "no significant shift" and v["status"] != "not applicable"}

    logging.info("")
    logging.info("=== Feature drift summary ===")
    if flagged:
        for feat, res in sorted(flagged.items(), key=lambda x: -x[1]["psi"]):
            logging.info(f"  {feat}: PSI={res['psi']:.4f} ({res['status']})")
    else:
        logging.info("  No features flagged for moderate or significant drift.")

    prediction_drift = None
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        X_ref = reference.drop(columns=[target_col])
        X_cur = current.drop(columns=[target_col])
        common_cols = [c for c in X_ref.columns if c in X_cur.columns]

        ref_proba = model.predict_proba(X_ref[common_cols])[:, 1]
        cur_proba = model.predict_proba(X_cur[common_cols])[:, 1]
        pred_psi = calculate_psi(ref_proba, cur_proba)
        prediction_drift = {"psi": pred_psi, "status": classify_psi(pred_psi)}

        logging.info("")
        logging.info("=== Prediction (output) drift ===")
        logging.info(f"  Predicted probability PSI={pred_psi:.4f} ({prediction_drift['status']})")
    else:
        logging.info("")
        logging.info("(No --model provided or file not found -- skipping prediction-level drift check.)")

    report = {
        "reference_file": reference_path,
        "current_file": current_path,
        "reference_n": len(reference),
        "current_n": len(current),
        "feature_drift": feature_results,
        "flagged_features": list(flagged.keys()),
        "prediction_drift": prediction_drift,
        "thresholds": {"moderate": PSI_MODERATE_THRESHOLD, "significant": PSI_SIGNIFICANT_THRESHOLD},
    }

    output_dir = os.path.dirname(current_path) or "."
    report_path = os.path.join(output_dir, "drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("")
    logging.info(f"Full report saved to {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check data and prediction drift using PSI.")
    parser.add_argument("--reference", required=True, help="Path to the reference dataset (e.g. train_final.csv)")
    parser.add_argument("--current", required=True, help="Path to the current dataset to check for drift (e.g. a recent scoring batch)")
    parser.add_argument("--target-col", default=None, help="Target column name to exclude from feature comparison (auto-detected if omitted: TARGET or 'default payment_next_month')")
    parser.add_argument("--model", default=None, help="Optional path to a trained .joblib model, to also check prediction-level drift")
    args = parser.parse_args()

    target_col = args.target_col
    if target_col is None:
        # Auto-detect based on which of the two known target column names is present.
        probe = pd.read_csv(args.reference, nrows=1)
        if "TARGET" in probe.columns:
            target_col = "TARGET"
        elif "default payment_next_month" in probe.columns:
            target_col = "default payment_next_month"
        else:
            raise ValueError("Could not auto-detect the target column -- pass --target-col explicitly.")

    run(args.reference, args.current, target_col, args.model)
