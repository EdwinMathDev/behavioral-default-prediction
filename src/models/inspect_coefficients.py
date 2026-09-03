"""
inspect_coefficients.py
=========================

Quick, cheap sanity check — Home Credit Default Risk (v2).

Responsibility
--------------
Before investing effort in more complex feature engineering (interaction
terms, segmented models) to address the "diluted signal" hypothesis
(behavioral features like CC_AVG_UTILIZATION are NaN-imputed with the
train median for the ~72% of clients with no credit card history, which
could be masking their real signal), check something cheap first: does the
trained Logistic Regression already give meaningful weight to the
HAS_*_HISTORY flags?

If it does, the model already has SOME way to distinguish "has history,
average utilization" from "no history, imputed to average utilization" —
weakening the diluted-signal hypothesis. If those flags have near-zero
coefficients, that's evidence the hypothesis deserves a deeper look.

This does NOT retrain anything — it just inspects the already-trained
logreg_baseline_v2.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = "models/artifacts"

model = joblib.load(os.path.join(ARTIFACTS_DIR, "logreg_baseline_v2.joblib"))
feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns_v2.joblib"))

coefs = pd.Series(model.coef_[0], index=feature_columns).sort_values(key=np.abs, ascending=False)

print("Top 15 coefficients by absolute magnitude (standardized scale):")
print(coefs.head(15))
print()

history_flags = [c for c in feature_columns if c.startswith("HAS_") and c.endswith("_HISTORY")]
print("Coefficients for the HAS_*_HISTORY flags specifically:")
for flag in history_flags:
    rank = list(coefs.index).index(flag) + 1
    print(f"  {flag}: coef={coefs[flag]:+.4f}  (rank {rank} of {len(coefs)} by |magnitude|)")
