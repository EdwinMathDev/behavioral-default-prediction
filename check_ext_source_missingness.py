"""
check_ext_source_missingness.py
==================================

Quick, cheap check — Home Credit Default Risk (v2).

Responsibility
--------------
Before deciding whether to keep EXT_SOURCE_1/2/3 as production features
(they contribute +0.0507 AUC per ablation_check_ext_source_v2.py, by far
the strongest signal in the model), check something that matters more for
a production decision than raw AUC: how often are they actually available?

If a "vendor score" is missing for a large share of applicants, leaning on
it heavily creates a real production risk — the model becomes only as
reliable as a third-party data feed's coverage, and missingness itself may
correlate with risk in ways that deserve scrutiny (same kind of pattern
already found with HAS_BUREAU_HISTORY being counterintuitively predictive).

This does not retrain anything — it only inspects raw application_train.csv.
"""

import pandas as pd

df = pd.read_csv("data/raw/home_credit/application_train.csv", usecols=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "TARGET"])

for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
    missing_pct = df[col].isna().mean()
    print(f"{col}: {missing_pct:.1%} missing")

print()
all_missing = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isna().all(axis=1)
print(f"Clients missing ALL THREE EXT_SOURCE columns: {all_missing.sum():,} ({all_missing.mean():.2%})")

print()
print("Default rate: clients missing ALL THREE vs. clients with at least one:")
print(df.groupby(all_missing)["TARGET"].agg(["mean", "count"]))
