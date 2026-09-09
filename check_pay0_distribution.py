"""
check_pay0_distribution.py
=============================

One-off diagnostic script -- NOT part of the pipeline.

Purpose: compare the real value frequencies of PAY_0 between train_final.csv
and test_final.csv, to determine whether the PSI=0.2884 "significant drift"
flagged by check_data_drift.py reflects a genuine difference in the data,
or an artifact of quantile-based binning applied to a discrete variable.
"""

import pandas as pd

train = pd.read_csv("data/features/train_final.csv")
test = pd.read_csv("data/features/test_final.csv")

train_freq = train["PAY_0"].value_counts(normalize=True).sort_index()
test_freq = test["PAY_0"].value_counts(normalize=True).sort_index()

comparison = pd.DataFrame({"train": train_freq, "test": test_freq})
comparison["abs_diff"] = (comparison["train"] - comparison["test"]).abs()

print(comparison)
print()
print(f"Sum of absolute differences: {comparison['abs_diff'].sum():.4f}")
