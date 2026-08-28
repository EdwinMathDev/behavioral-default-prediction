"""
compare_baseline_vs_challenger.py
===================================

Statistical comparison — Credit Risk Engine / behavioral-default-prediction.

Responsibility
--------------
The baseline (Logistic Regression) and the challenger (XGBoost) were
both cross-validated with the SAME StratifiedKFold(random_state=42),
so fold i in one report corresponds to the exact same train/test
split as fold i in the other. This makes the comparison PAIRED:
instead of comparing mean +/- std separately, we compute the
per-fold difference (challenger - baseline) and run a paired t-test
on those differences.

A paired t-test is much more sensitive than comparing two
independent-looking distributions, because it cancels out the
fold-to-fold variance that affects both models equally (e.g. a fold
that happens to be "harder" for everyone), isolating only the
variance that comes from the model choice itself.

Decision rule
--------------
If p-value < 0.05, the challenger's improvement is unlikely to be due
to chance from the fold split, and can be considered a real
improvement over the baseline (still to be weighed against the loss
of interpretability). If p-value >= 0.05, the observed difference is
consistent with random fold-to-fold variation, and per the project's
own documented principle (train_baseline.py docstring), the baseline
should be preferred: same predictive power, full interpretability.

Input
-----
    models/artifacts/baseline_cross_validation_report.json
    models/artifacts/challenger_cross_validation_report.json

Output
------
    Printed comparison table + paired t-test results for AUC-ROC and KS.
"""

import os
import json
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ARTIFACTS_DIR = "models/artifacts"


def load_report(filename: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


def paired_ttest(baseline_values: list, challenger_values: list, metric_name: str) -> dict:
    baseline_arr = np.array(baseline_values)
    challenger_arr = np.array(challenger_values)
    diffs = challenger_arr - baseline_arr

    t_stat, p_value = stats.ttest_rel(challenger_arr, baseline_arr)

    result = {
        "metric": metric_name,
        "baseline_mean": float(baseline_arr.mean()),
        "challenger_mean": float(challenger_arr.mean()),
        "mean_diff": float(diffs.mean()),
        "diffs_per_fold": diffs.tolist(),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }
    return result


def run():
    baseline_report = load_report("baseline_cross_validation_report.json")
    challenger_report = load_report("challenger_cross_validation_report.json")

    baseline_folds = baseline_report["folds"]
    challenger_folds = challenger_report["folds"]

    if len(baseline_folds) != len(challenger_folds):
        raise ValueError(
            f"Los reportes tienen distinto numero de folds "
            f"(baseline={len(baseline_folds)}, challenger={len(challenger_folds)}). "
            f"No se puede hacer una comparacion pareada valida."
        )

    baseline_auc = [f["auc_roc"] for f in baseline_folds]
    challenger_auc = [f["auc_roc"] for f in challenger_folds]
    baseline_ks = [f["ks_statistic"] for f in baseline_folds]
    challenger_ks = [f["ks_statistic"] for f in challenger_folds]

    logging.info("Fold-by-fold comparison (challenger - baseline):")
    for i in range(len(baseline_folds)):
        auc_diff = challenger_auc[i] - baseline_auc[i]
        ks_diff = challenger_ks[i] - baseline_ks[i]
        logging.info(
            f"  Fold {i+1}: AUC diff = {auc_diff:+.4f} | KS diff = {ks_diff:+.4f}"
        )

    auc_result = paired_ttest(baseline_auc, challenger_auc, "auc_roc")
    ks_result = paired_ttest(baseline_ks, challenger_ks, "ks_statistic")

    logging.info("")
    logging.info("=== AUC-ROC ===")
    logging.info(f"  Baseline:  {auc_result['baseline_mean']:.4f}")
    logging.info(f"  Challenger: {auc_result['challenger_mean']:.4f}")
    logging.info(f"  Diferencia media: {auc_result['mean_diff']:+.4f}")
    logging.info(f"  t-statistic: {auc_result['t_statistic']:.4f} | p-value: {auc_result['p_value']:.4f}")
    logging.info(f"  Significativo al 5%: {auc_result['significant_at_0.05']}")

    logging.info("")
    logging.info("=== KS statistic ===")
    logging.info(f"  Baseline:  {ks_result['baseline_mean']:.4f}")
    logging.info(f"  Challenger: {ks_result['challenger_mean']:.4f}")
    logging.info(f"  Diferencia media: {ks_result['mean_diff']:+.4f}")
    logging.info(f"  t-statistic: {ks_result['t_statistic']:.4f} | p-value: {ks_result['p_value']:.4f}")
    logging.info(f"  Significativo al 5%: {ks_result['significant_at_0.05']}")

    logging.info("")
    if not auc_result["significant_at_0.05"] and not ks_result["significant_at_0.05"]:
        logging.info(
            "CONCLUSION: Ninguna diferencia es estadisticamente significativa. "
            "Segun el criterio documentado en train_baseline.py, se recomienda "
            "quedarse con la Regresion Logistica (mismo poder predictivo, "
            "totalmente interpretable)."
        )
    else:
        logging.info(
            "CONCLUSION: Al menos una metrica muestra una mejora estadisticamente "
            "significativa del challenger. Evaluar si justifica la perdida de "
            "interpretabilidad para el caso de uso."
        )

    output = {"auc_roc": auc_result, "ks_statistic": ks_result}
    report_path = os.path.join(ARTIFACTS_DIR, "baseline_vs_challenger_ttest.json")
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Reporte guardado en {report_path}")

    return output


if __name__ == "__main__":
    run()
