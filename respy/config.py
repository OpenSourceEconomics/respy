"""General configuration for respy."""
from pathlib import Path

import numpy as np

# Obtain the root directory of the package. Do not import respy which creates a circular
# import.
ROOT_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = ROOT_DIR / "tests" / "resources"

# Set maximum numbers to 1e200 and log(1e200) = 460.
MAX_FLOAT = 1e200
MIN_FLOAT = -MAX_FLOAT
MAX_LOG_FLOAT = 460
MIN_LOG_FLOAT = -MAX_LOG_FLOAT

INADMISSIBILITY_PENALTY = -400_000
"""int : Penalty for choosing an inadmissible state.

The penalty is applied to the non-pecuniary reward of choice which cannot be taken.

See Also
--------
respy.pre_processing.model_processing._parse_inadmissibility_penalty
respy.state_space._create_is_inadmissible
respy.solve._create_choice_rewards

"""


# Everything for the indexer.
INDEXER_DTYPE = np.int32
"""numpy.dtype : Data type for the entries in the state space indexer."""
INDEXER_INVALID_INDEX = np.iinfo(INDEXER_DTYPE).min
"""int : Identifier for invalid states.

Every valid state has a unique number which is stored in the state space indexer at the
correct position. Invalid entries in the indexer are filled with
:data:`INDEXER_INVALID_INDEX` which is the most negative value for
:data:`INDEXER_DTYPE`. Using the invalid value as an index likely raises an
:class:`IndexError` as negative indices cannot exceed the length of the indexed array
dimension.

"""

# Some assert functions take rtol instead of decimals
TOL_REGRESSION_TESTS = 1e-10

SEED_STARTUP_ITERATION_GAP = 100

DEFAULT_OPTIONS = {
    "estimation_draws": 200,
    "estimation_seed": 1,
    "estimation_tau": 500,
    "interpolation_points": -1,
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
