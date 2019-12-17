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
MAX_FLOAT = 1e300

# Some assert functions take rtol instead of decimals
TOL_REGRESSION_TESTS = 1e-10

# Interpolation
INADMISSIBILITY_PENALTY = -400_000

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
    "solution_draws": 200,
    "solution_seed": 3,
    "core_state_space_filters": [],
    "inadmissible_states": {},
    "monte_carlo_sequence": "sobol",
}

KEANE_WOLPIN_1994_MODELS = [f"kw_94_{suffix}" for suffix in ["one", "two", "three"]]
KEANE_WOLPIN_1997_MODELS = ["kw_97_basic", "kw_97_extended"]
KEANE_WOLPIN_2000_MODELS = ["kw_2000"]
ROBINSON_CRUSOE_MODELS = ["robinson_crusoe_basic", "robinson_crusoe_extended"]

EXAMPLE_MODELS = (
    KEANE_WOLPIN_1994_MODELS
    + KEANE_WOLPIN_1997_MODELS
    + KEANE_WOLPIN_2000_MODELS
    + ROBINSON_CRUSOE_MODELS
)
