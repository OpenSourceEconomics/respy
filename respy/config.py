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
    "not_exp_a_lagged": "exp_a > 0 and lagged_choice != 'a'",
    "not_exp_b_lagged": "exp_b > 0 and lagged_choice != 'b'",
    "work_a_lagged": "lagged_choice == 'a'",
    "work_b_lagged": "lagged_choice == 'b'",
    "edu_lagged": "lagged_choice == 'edu'",
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
    "at_least_ten_years_edu": "10 <= exp_edu",
}
"""dict: Dictionary containing specification of covariates.

The keys of the dictionary are used as column names and must correspond to the parameter
value in the parameter specification. The values are strings passed to ``pd.eval``.

"""

BASE_CORE_STATE_SPACE_FILTERS = [
    # In periods > 0, if agents accumulated experience only in one choice, lagged choice
    # cannot be different.
    "period > 0 and exp_{i} == period and lagged_choice != '{i}'",
    # In periods > 0, if agents always accumulated experience, lagged choice cannot be
    # non-experience choice.
    "period > 0 and exp_a + exp_b + exp_edu == period and lagged_choice == '{j}'",
    # In periods > 0, if agents accumulated no years of schooling, lagged choice cannot
    # be school.
    "period > 0 and lagged_choice == 'edu' and exp_edu == 0",
    # If experience in choice 0 and 1 are zero, lagged choice cannot be this choice.
    "lagged_choice == '{k}' and exp_{k} == 0",
    # In period 0, agents cannot choose occupation a or b.
    "period == 0 and lagged_choice == '{k}'",
]
"""list: Contains filters for the state space.

These formulas are applied to the core state space which abstracts from initial
experiences and uses only the maximum range between initial experiences and maximum
experiences.

See also
--------
respy.state_space._filter_core_state_space : Filters the core state space.
respy.state_space._create_core_state_space : Creates the core state space.

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
    "core_state_space_filters": BASE_CORE_STATE_SPACE_FILTERS,
}

KEANE_WOLPIN_1994_MODELS = [
    f"kw_data_{suffix}"
    for suffix in ["one", "one_initial", "one_types", "two", "three"]
] + ["reliability_short"]

KEANE_WOLPIN_1997_MODELS = ["kw_97_base", "kw_97_extended"]

EXAMPLE_MODELS = KEANE_WOLPIN_1994_MODELS + KEANE_WOLPIN_1997_MODELS
