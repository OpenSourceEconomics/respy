""" Module for program constants used across the RESPY package. This is
aligned with the constants from the FORTRAN implementation.
"""

import os

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
OPT_EST_PYTH = ['SCIPY-BFGS', 'SCIPY-POWELL']
OPT_EST_FORT = ['FORT-NEWUOA', 'FORT-BFGS']

# TODO: THIS is only temporary as I slowly channel the info through the
# interfacse.
optimizer_options = None
dfunc_eps = None