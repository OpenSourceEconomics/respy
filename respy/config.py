"""General configuration for respy."""
from pathlib import Path

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
# Some assert functions take rtol instead of decimals
TOL = 10 ** -DECIMALS

# Interpolation
INADMISSIBILITY_PENALTY = -400000

IS_DEBUG = False

BASE_COVARIATES = {
    "not_exp_a_lagged": "exp_a > 0 and lagged_choice_1 != 'a'",
    "not_exp_b_lagged": "exp_b > 0 and lagged_choice_1 != 'b'",
    "work_a_lagged": "lagged_choice_1 == 'a'",
    "work_b_lagged": "lagged_choice_1 == 'b'",
    "edu_lagged": "lagged_choice_1 == 'edu'",
    "not_any_exp_a": "exp_a == 0",
    "not_any_exp_b": "exp_b == 0",
    "any_exp_a": "exp_a > 0",
    "any_exp_b": "exp_b > 0",
    "hs_graduate": "exp_edu >= 12",
    "co_graduate": "exp_edu >= 16",
    "is_return_not_high_school": "~edu_lagged and ~hs_graduate",
    "is_return_high_school": "~edu_lagged and hs_graduate",
    "is_minor": "period < 2",
    "is_young_adult": "2 <= period <= 4",
    "is_adult": "5 <= period",
    "constant": "1",
    "exp_a_square": "exp_a ** 2 / 100",
    "exp_b_square": "exp_b ** 2 / 100",
    "up_to_nine_years_edu": "exp_edu <= 9",
    "at_least_ten_years_edu": "exp_edu >= 10",
    "edu_ten": "exp_edu == 10",
}
"""dict: Dictionary containing specification of covariates.

The keys of the dictionary are used as column names and must correspond to the parameter
value in the parameter specification. The values are strings passed to ``pd.eval``.

"""

BASE_INADMISSIBLE_STATES = {"edu": "exp_edu == @max_exp_edu"}

DEFAULT_OPTIONS = {
    "choices": {"edu": {"max": 20, "start": [10], "lagged": [1], "share": [1]}},
    "estimation_draws": 200,
    "estimation_seed": 1,
    "estimation_tau": 500,
    "interpolation_points": -1,
    "n_periods": 40,
    "simulation_agents": 1000,
    "simulation_seed": 2,
    "solution_draws": 500,
    "solution_seed": 3,
    "covariates": BASE_COVARIATES,
    "inadmissible_states": BASE_INADMISSIBLE_STATES,
    "core_state_space_filters": [],
}

KEANE_WOLPIN_1994_MODELS = [f"kw_94_{suffix}" for suffix in ["one", "two", "three"]]
KEANE_WOLPIN_1997_MODELS = ["kw_97_base", "kw_97_extended"]

EXAMPLE_MODELS = KEANE_WOLPIN_1994_MODELS + KEANE_WOLPIN_1997_MODELS
