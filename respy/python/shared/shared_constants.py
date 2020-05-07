""" Module for program constants used across the RESPY package. This is
aligned with the constants from the FORTRAN implementation.
"""

import os
from pathlib import Path

# Obtain the root directory of the package
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/python/shared', '')

# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR + '/tests'
TEST_RESOURCES_DIR = ROOT_DIR + '/tests/resources'

# Directory with the FORTRAN resources
EXEC_DIR = ROOT_DIR + '/fortran/bin'

LARGE_FLOAT = 1.0e8
HUGE_FLOAT = 1.0e20
SMALL_FLOAT = 1e-5
TINY_FLOAT = 1.0e-8

# Interpolation
INADMISSIBILITY_PENALTY = -40000.00

# Missing values. These allow to aline the treatment of missing values across
# implementations. There is no NAN available in FORTRAN.
MISSING_INT = -99
MISSING_FLOAT = -99.00

# Flag that indicate whether the FORTRAN executables are available.
IS_PARALLEL = os.path.exists(EXEC_DIR + '/resfort_parallel_master')
IS_FORTRAN = os.path.exists(EXEC_DIR + '/resfort_scalar')
if not IS_FORTRAN:
    assert (not IS_PARALLEL)

# Each implementation has its own set of optimizers available.
OPTIMIZERS_PYTH = ['SCIPY-BFGS', 'SCIPY-POWELL']
OPTIMIZERS_FORT = ['FORT-NEWUOA', 'FORT-BFGS']


example_dir = Path(__file__).resolve().parents[3] / "example"

MODEL_TO_INI = {
    "kw_data_one": str(example_dir / "kw_data_one.ini"),
    "kw_data_two": str(example_dir / "kw_data_two.ini"),
    "kw_data_three": str(example_dir / "kw_data_three.ini"),
    "example": str(example_dir / "example.ini"),
}


INDEX_TUPLES = [
        ('delta', 'delta'),
        ('wage_a', 'constant'),
        ('wage_a', 'exp_edu'),
        ('wage_a', 'exp_a'),
        ('wage_a', 'exp_a_square'),
        ('wage_a', 'exp_b'),
        ('wage_a', 'exp_b_square'),
        ('wage_b', 'constant'),
        ('wage_b', 'exp_edu'),
        ('wage_b', 'exp_a'),
        ('wage_b', 'exp_a_square'),
        ('wage_b', 'exp_b'),
        ('wage_b', 'exp_b_square'),
        ('nonpec_edu', 'constant'),
        ('nonpec_edu', 'at_least_twelve_exp_edu'),
        ('nonpec_edu', 'not_edu_last_period'),
        ('nonpec_home', 'constant'),
        ('shocks_chol', 'chol_a'),
        ('shocks_chol', 'chol_b_a'),
        ('shocks_chol', 'chol_b'),
        ('shocks_chol', 'chol_edu_a'),
        ('shocks_chol', 'chol_edu_b'),
        ('shocks_chol', 'chol_edu'),
        ('shocks_chol', 'chol_home_a'),
        ('shocks_chol', 'chol_home_b'),
        ('shocks_chol', 'chol_home_edu'),
        ('shocks_chol', 'chol_home'),
    ]


DATA_COLUMNS = [
    "Identifier",
    "Period",
    "Choice",
    "Wage",
    "Experience_A",
    "Experience_B",
    "Experience_Edu",
    "Lagged Schooling",
]
