"""General configuration for respy."""
from pathlib import Path

import numpy as np

# Obtain the root directory of the package. Do not import respy which creates a circular
# import.
ROOT_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = ROOT_DIR / "tests" / "resources"

HUGE_FLOAT = 1e20
TINY_FLOAT = 1e-8
PRINT_FLOAT = 1e10

# Number of decimals that are compared for tests This is currently only used in
# regression tests.
DECIMALS = 6
# Some assert fucntions take rtol instead of decimals
TOL = 10 ** -DECIMALS

# Interpolation
INADMISSIBILITY_PENALTY = -400000

IS_DEBUG = False

# Labels for columns in a dataset as well as the formatters.
DATA_LABELS_EST = [
    "Identifier",
    "Period",
    "Choice",
    "Wage",
    "Experience_A",
    "Experience_B",
    "Years_Schooling",
    "Lagged_Choice",
]

# There is additional information available in a simulated dataset.
DATA_LABELS_SIM = DATA_LABELS_EST + [
    "Type",
    "Total_Reward_1",
    "Total_Reward_2",
    "Total_Reward_3",
    "Total_Reward_4",
    "Systematic_Reward_1",
    "Systematic_Reward_2",
    "Systematic_Reward_3",
    "Systematic_Reward_4",
    "Shock_Reward_1",
    "Shock_Reward_2",
    "Shock_Reward_3",
    "Shock_Reward_4",
    "Discount_Rate",
    "Immediate_Reward_1",
    "Immediate_Reward_2",
    "Immediate_Reward_3",
    "Immediate_Reward_4",
]

DATA_FORMATS_EST = {
    col: (np.float if col == "Wage" else np.int) for col in DATA_LABELS_EST
}
DATA_FORMATS_SIM = {
    col: (np.int if col == "Type" else np.float) for col in DATA_LABELS_SIM
}
DATA_FORMATS_SIM = {**DATA_FORMATS_SIM, **DATA_FORMATS_EST}

EXAMPLE_MODELS = [
    f"kw_data_{suffix}"
    for suffix in ["one", "one_initial", "one_types", "two", "three"]
] + ["reliability_short"]
