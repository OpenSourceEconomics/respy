import json
import os
import sys
import warnings

import pandas as pd
import pytest

from respy.config import IS_DEBUG
from respy.config import ROOT_DIR
from respy.config import TEST_RESOURCES_DIR

# We only maintain the code base for Python >= 3.6
assert sys.version_info[:2] >= (3, 6)


# We want to turn off the nuisance warnings while in production.
if not IS_DEBUG:
    warnings.simplefilter(action="ignore", category=FutureWarning)

__version__ = "1.2.1"


EXAMPLE_MODELS = [
    TEST_RESOURCES_DIR / f"kw_data_{suffix}"
    for suffix in ["one", "one_initial", "one_types", "two", "three"]
] + [TEST_RESOURCES_DIR / "reliability_short"]


def test(opt=None):
    """Run basic tests of the package."""
    current_directory = os.getcwd()
    os.chdir(ROOT_DIR)
    pytest.main(opt)
    os.chdir(current_directory)


def get_example_model(model):
    assert model in EXAMPLE_MODELS

    options_spec = json.loads((TEST_RESOURCES_DIR / f"{model}.json").read_text())
    params_spec = pd.read_csv(TEST_RESOURCES_DIR / f"{model}.csv")

    return params_spec, options_spec
