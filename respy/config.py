"""General configuration for respy."""
from pathlib import Path

# Obtain the root directory of the package. Do not import respy which creates a circular
# import.
ROOT_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = ROOT_DIR / "tests" / "resources"

MAX_FLOAT = 1e300
MIN_FLOAT = -1e300
MAX_LOG_FLOAT = 700
MIN_LOG_FLOAT = -700

# Number of decimals that are compared for tests This is currently only used in
# regression tests.
DECIMALS = 6
# Some assert functions take rtol instead of decimals
TOL = 10 ** -DECIMALS

# Interpolation
INADMISSIBILITY_PENALTY = -400000

SEED_STARTUP_ITERATION_GAP = 100

IS_DEBUG = False

DEFAULT_OPTIONS = {
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
KEANE_WOLPIN_1997_MODELS = ["kw_97_basic", "kw_97_extended"]

EXAMPLE_MODELS = KEANE_WOLPIN_1994_MODELS + KEANE_WOLPIN_1997_MODELS
