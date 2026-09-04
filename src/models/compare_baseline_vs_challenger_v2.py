"""
compare_baseline_vs_challenger_v2.py
=======================================

Statistical comparison — Home Credit Default Risk (v2).

Responsibility
--------------
Same method as v1's compare_baseline_vs_challenger.py: baseline and
challenger were cross-validated on the SAME StratifiedKFold(random_state=42),
so fold i in one report is the exact same split as fold i in the other.
Paired t-test on the per-fold differences, not a naive comparison of means.

Decision rule
--------------
If p-value < 0.05, the challenger's improvement is unlikely to be due to
fold-to-fold chance. If p-value >= 0.05, per this project's established
principle (train_baseline_v2.py docstring, mirroring v1's), the baseline
should be preferred: same predictive power, full interpretability.

Input
-----
    models/artifacts/baseline_cross_validation_report_v2.json
    models/artifacts/challenger_cross_validation_report_v2.json

Output
------
    Printed comparison + models/artifacts/baseline_vs_challenger_v2_ttest.json
"""

import os
import json
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARTIFACTS_DIR = "models/artifacts"


def load_report(filename: str) -> dict:
    with open(os.path.join(ARTIFACTS_DIR, filename), "r") as f:
        return json.load(f)


def paired_ttest(baseline_values, challenger_values, metric_name: str) -> dict:
    baseline_arr = np.array(baseline_values)
    challenger_arr = np.array(challenger_values)
    diffs = challenger_arr - baseline_arr

    t_stat, p_value = stats.ttest_rel(challenger_arr, baseline_arr)

    return {
        "metric": metric_name,
        "baseline_mean": float(baseline_arr.mean()),
        "challenger_mean": float(challenger_arr.mean()),
        "mean_diff": float(diffs.mean()),
        "diffs_per_fold": diffs.tolist(),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def run():
    baseline_report = load_report("baseline_cross_validation_report_v2.json")
    challenger_report = load_report("challenger_cross_validation_report_v2.json")

    baseline_folds = baseline_report["folds"]
    challenger_folds = challenger_report["folds"]

    if len(baseline_folds) != len(challenger_folds):
        raise ValueError("Fold count mismatch — cannot run a valid paired comparison.")

    baseline_auc = [f["auc_roc"] for f in baseline_folds]
    challenger_auc = [f["auc_roc"] for f in challenger_folds]
    baseline_ks = [f["ks_statistic"] for f in baseline_folds]
    challenger_ks = [f["ks_statistic"] for f in challenger_folds]

    logging.info("Fold-by-fold comparison (challenger - baseline):")
    for i in range(len(baseline_folds)):
        logging.info(f"  Fold {i+1}: AUC diff = {challenger_auc[i]-baseline_auc[i]:+.4f} "
                      f"| KS diff = {challenger_ks[i]-baseline_ks[i]:+.4f}")

    auc_result = paired_ttest(baseline_auc, challenger_auc, "auc_roc")
    ks_result = paired_ttest(baseline_ks, challenger_ks, "ks_statistic")

    logging.info("")
    logging.info("=== AUC-ROC ===")
    logging.info(f"  Baseline (LogReg v2):   {auc_result['baseline_mean']:.4f}")
    logging.info(f"  Challenger (XGBoost v2): {auc_result['challenger_mean']:.4f}")
    logging.info(f"  Mean diff: {auc_result['mean_diff']:+.4f} | p-value: {auc_result['p_value']:.4f} "
                  f"| Significant: {auc_result['significant_at_0.05']}")

    logging.info("")
    logging.info("=== KS statistic ===")
    logging.info(f"  Baseline (LogReg v2):   {ks_result['baseline_mean']:.4f}")
    logging.info(f"  Challenger (XGBoost v2): {ks_result['challenger_mean']:.4f}")
    logging.info(f"  Mean diff: {ks_result['mean_diff']:+.4f} | p-value: {ks_result['p_value']:.4f} "
                  f"| Significant: {ks_result['significant_at_0.05']}")

    logging.info("")
    if not auc_result["significant_at_0.05"] and not ks_result["significant_at_0.05"]:
        logging.info("CONCLUSION: No significant difference. Keep Logistic Regression (v2) — "
                      "same predictive power, full interpretability.")
    else:
        logging.info("CONCLUSION: At least one metric shows a statistically significant "
                      "improvement for the challenger. Evaluate whether it justifies the "
                      "loss of interpretability, and whether it closes the gap with v1.")

    output = {"auc_roc": auc_result, "ks_statistic": ks_result}
    with open(os.path.join(ARTIFACTS_DIR, "baseline_vs_challenger_v2_ttest.json"), "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Report saved to {ARTIFACTS_DIR}/baseline_vs_challenger_v2_ttest.json")

    return output


if __name__ == "__main__":
    run()
