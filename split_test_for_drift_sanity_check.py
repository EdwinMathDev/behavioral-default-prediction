"""
split_test_for_drift_sanity_check.py
=======================================

One-off diagnostic script -- NOT part of the pipeline.

Purpose: split test_final.csv (real, non-SMOTE data) into two random
halves, to sanity-check check_data_drift.py without the train_final.csv
SMOTE-contamination issue. Since both halves come from the exact same
real population, PSI should be near-zero everywhere -- confirming the
tool itself works correctly once given a valid reference.
"""

import pandas as pd

test = pd.read_csv("data/features/test_final.csv")
shuffled = test.sample(frac=1.0, random_state=7).reset_index(drop=True)

half_a = shuffled.iloc[: len(shuffled) // 2]
half_b = shuffled.iloc[len(shuffled) // 2 :]

half_a.to_csv("data/features/test_final_half_a.csv", index=False)
half_b.to_csv("data/features/test_final_half_b.csv", index=False)

print(f"Half A: {len(half_a):,} rows -> data/features/test_final_half_a.csv")
print(f"Half B: {len(half_b):,} rows -> data/features/test_final_half_b.csv")
