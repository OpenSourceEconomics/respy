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
    "core_state_space_filters": [],
    "inadmissible_states": {},
}

KEANE_WOLPIN_1994_MODELS = [f"kw_94_{suffix}" for suffix in ["one", "two", "three"]]
KEANE_WOLPIN_1997_MODELS = ["kw_97_base", "kw_97_extended"]

EXAMPLE_MODELS = KEANE_WOLPIN_1994_MODELS + KEANE_WOLPIN_1997_MODELS
