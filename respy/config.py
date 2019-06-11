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

BASE_COVARIATES = {
    # Experience in A or B, but not in the last period.
    "not_exp_a_lagged": "(exp_a > 0) & (choice_lagged != 0)",
    "not_exp_b_lagged": "(exp_b > 0) & (choice_lagged != 1)",
    # Last occupation was A, B, or education.
    "work_a_lagged": "choice_lagged == 0",
    "work_b_lagged": "choice_lagged == 1",
    "edu_lagged": "choice_lagged == 2",
    # No experience in A or B.
    "not_any_exp_a": "exp_a == 0",
    "not_any_exp_b": "exp_b == 0",
    # Any experience in A or B.
    "any_exp_a": "exp_a > 0",
    "any_exp_b": "exp_b > 0",
    # High school or college graduate.
    "hs_graduate": "exp_edu >= 12",
    "co_graduate": "exp_edu >= 16",
    # Was not in school last period and is/is not high school graduate.
    "is_return_not_high_school": "~edu_lagged & ~hs_graduate",
    "is_return_high_school": "~edu_lagged & hs_graduate",
    # Define age groups.
    "is_minor": "period < 2",
    "is_young_adult": "2 <= period <= 4",
    "is_adult": "5 <= period",
    # Constant.
    "constant": "1",
    # Squared experience in sectors.
    "exp_a_square": "exp_a ** 2 / 100",
    "exp_b_square": "exp_b ** 2 / 100",
}
"""dict: Dictionary containing specification of covariates.

The keys of the dictionary are used as column names and must correspond to the parameter
value in the parameter specification. The values are strings passed to ``pandas.eval``.

"""

BASE_STATE_SPACE_FILTERS = [
    # In period 0, agents cannot choose occupation a or b.
    "(period == 0) & ((lagged_choice == 0) | (lagged_choice == 1))",
    # In periods > 0, if agents accumulated experience only in one sector, lagged choice
    # cannot be different.
    "(period > 0) & (exp_{i} - @initial_exp[{i}] == period) & (lagged_choice != {i})",
    # In periods > 0, if agents always accumulated experience, lagged choice cannot be
    # non-experience sector.
    "(period > 0) & (exp_0 + exp_1 + exp_2 - @initial_exp[2] == period) "
    "& (lagged_choice == {j})",
    # In periods > 0, if agents accumulated no years of schooling, lagged choice cannot
    # be school.
    "(period > 0) & (lagged_choice == 2) & (exp_2 == @initial_exp[2])",
    # If experience in sector 0 and 1 are zero, lagged choice cannot be this sector.
    "(lagged_choice == 0) & (exp_0 == 0)",
    "(lagged_choice == 1) & (exp_1 == 0)",
]
"""list: Contains filters for the state space.

TODO: Check for collinear restrictions.

"""

BASE_RESTRICTIONS = {
    "a": "False",
    "b": "False",
    "edu": "exp_edu == education_max",
    "home": "False",
}

DEFAULT_OPTIONS = {
    "education_lagged": [1],
    "education_start": [10],
    "education_share": [1],
    "education_max": 20,
    "estimation_draws": 200,
    "estimation_seed": 1,
    "estimation_tau": 500,
    "interpolation_points": -1,
    "num_periods": 40,
    "simulation_agents": 1000,
    "simulation_seed": 2,
    "solution_draws": 500,
    "solution_seed": 3,
    "covariates": BASE_COVARIATES,
    "inadmissible_states": BASE_RESTRICTIONS,
    "state_space_filters": BASE_STATE_SPACE_FILTERS,
}

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
    "Nonpecuniary_Rewards_0",
    "Nonpecuniary_Rewards_1",
    "Nonpecuniary_Rewards_2",
    "Nonpecuniary_Rewards_3",
    "Wages_0",
    "Wages_1",
    "Flow_Utility_0",
    "Flow_Utility_1",
    "Flow_Utility_2",
    "Flow_Utility_3",
    "Value_Function_0",
    "Value_Function_1",
    "Value_Function_2",
    "Value_Function_3",
    "Shock_Reward_0",
    "Shock_Reward_1",
    "Shock_Reward_2",
    "Shock_Reward_3",
    "Discount_Rate",
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
