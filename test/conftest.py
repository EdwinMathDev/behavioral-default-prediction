"""
conftest.py
=============

Shared pytest configuration for this project's test suite.

Responsibility
--------------
Some tests require artifacts that are intentionally NOT committed to git
(data/raw/*, data/features/*, models/artifacts/*.joblib — see .gitignore):
they're too large, or in v2's case, subject to Kaggle's redistribution
terms. On a developer's machine, after running the pipeline, these files
exist and the tests run for real. On a fresh CI checkout, they don't.

Rather than let those tests fail with a confusing "file not found" on
every CI run, this file automatically marks them as SKIPPED (not failed)
when their required artifacts are absent — the same pattern already used
inside test_config_v2.py, test_fairness_no_protected_attributes_v2.py, and
test_api_v2.py via pytest.skip(), applied here at the collection level so
it also covers the earlier v1 test files without needing to edit them.

This does NOT weaken the test suite: it only skips what genuinely cannot
be checked without the artifacts, and CI's summary will show exactly which
tests were skipped and why, rather than hiding the distinction inside a
blanket "all passed."
"""

import os
import pytest

# Maps a test file name to the artifact paths it needs to run for real.
# If ANY of a test file's required paths are missing, every test in that
# file is skipped (not failed).
ARTIFACT_REQUIREMENTS = {
    "test_config.py": [
        os.path.join("config", "model_config.json"),
    ],
    "test_config_v2.py": [
        os.path.join("config", "model_config_v2.json"),
    ],
    "test_fairness_no_protected_attributes.py": [
        os.path.join("models", "artifacts"),
    ],
    "test_fairness_no_protected_attributes_v2.py": [
        os.path.join("models", "artifacts"),
    ],
    "test_api_v2.py": [
        os.path.join("config", "model_config_v2.json"),
        os.path.join("data", "features", "home_credit_features.csv"),
        os.path.join("models", "artifacts"),
    ],
    "test_api_schema.py": [],  # pure Pydantic schema validation, no artifacts needed
    "test_build_features.py": [],  # synthetic data only
    "test_build_features_v2.py": [],  # synthetic data only
}


def _artifact_missing(paths):
    """A directory path is considered present if it exists AND has at
    least one file in it (an empty models/artifacts/ dir from a fresh
    clone still counts as 'missing')."""
    for p in paths:
        if not os.path.exists(p):
            return True
        if os.path.isdir(p) and not os.listdir(p):
            return True
    return False


def pytest_collection_modifyitems(config, items):
    for item in items:
        filename = os.path.basename(str(item.fspath))
        required = ARTIFACT_REQUIREMENTS.get(filename)
        if required and _artifact_missing(required):
            item.add_marker(pytest.mark.skip(
                reason=f"Requires local artifacts not present (expected: {required}). "
                       f"Run the pipeline locally (see README) to generate them."
            ))
