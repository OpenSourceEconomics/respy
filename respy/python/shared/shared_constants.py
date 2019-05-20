"""General configuration for respy."""
from pathlib import Path

import numpy as np

# Obtain the root directory of the package. Do not import respy which creates a circular
# import.
ROOT_DIR = Path(__file__).parents[2]

# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR / "tests"
TEST_RESOURCES_DIR = ROOT_DIR / "tests" / "resources"

MINISCULE_FLOAT = 1.0e-100
LARGE_FLOAT = 1.0e8
HUGE_FLOAT = 1.0e20
SMALL_FLOAT = 1e-5
TINY_FLOAT = 1.0e-8
PRINT_FLOAT = 1e10

# Number of decimals that are compared for tests
# This is currently only used in regression tests.
DECIMALS = 6
# Some assert fucntions take rtol instead of decimals
TOL = 10 ** -DECIMALS

# Interpolation
INADMISSIBILITY_PENALTY = -400000.00

# Missing values. These allow to aline the treatment of missing values across
# implementations. There is no NAN available in FORTRAN.
MISSING_INT = -99
MISSING_FLOAT = -99.00

IS_DEBUG = False

# Each implementation has its own set of optimizers available.
OPTIMIZERS = ["SCIPY-BFGS", "SCIPY-POWELL", "SCIPY-LBFGSB"]

# Labels for columns in a dataset as well as the formatters.
DATA_LABELS_EST = []
DATA_LABELS_EST += ["Identifier", "Period", "Choice", "Wage"]
DATA_LABELS_EST += ["Experience_A", "Experience_B", "Years_Schooling"]
DATA_LABELS_EST += ["Lagged_Choice"]

# There is additional information available in a simulated dataset.
DATA_LABELS_SIM = DATA_LABELS_EST[:]
DATA_LABELS_SIM += ["Type"]
DATA_LABELS_SIM += ["Total_Reward_1", "Total_Reward_2"]
DATA_LABELS_SIM += ["Total_Reward_3", "Total_Reward_4"]
DATA_LABELS_SIM += ["Systematic_Reward_1", "Systematic_Reward_2"]
DATA_LABELS_SIM += ["Systematic_Reward_3", "Systematic_Reward_4"]
DATA_LABELS_SIM += ["Shock_Reward_1", "Shock_Reward_2"]
DATA_LABELS_SIM += ["Shock_Reward_3", "Shock_Reward_4"]
DATA_LABELS_SIM += [
    "Discount_Rate",
    "General_Reward_1",
    "General_Reward_2",
    "Common_Reward",
]
DATA_LABELS_SIM += ["Immediate_Reward_1", "Immediate_Reward_2", "Immediate_Reward_3"]
DATA_LABELS_SIM += ["Immediate_Reward_4"]

DATA_FORMATS_EST = {}
for key_ in DATA_LABELS_EST:
    DATA_FORMATS_EST[key_] = np.int
    if key_ in ["Wage"]:
        DATA_FORMATS_EST[key_] = np.float

DATA_FORMATS_SIM = dict(DATA_FORMATS_EST)
for key_ in DATA_LABELS_SIM:
    if key_ in DATA_FORMATS_SIM.keys():
        continue
    elif key_ in ["Type"]:
        DATA_FORMATS_SIM[key_] = np.int
    else:
        DATA_FORMATS_SIM[key_] = np.float

# Set Numba configuration.
import numba  # noqa: E402

if IS_DEBUG:
    numba.config.WARNINGS = 1
