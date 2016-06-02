""" Module for program constants used across the RESPY package. This is
aligned with the constants from the FORTRAN implementation.
"""

# standard library
import os

# Obtain the root directory of the package
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/python/shared', '')

# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR + '/tests'
TEST_RESOURCES_DIR = ROOT_DIR + '/tests/resources'

# Directory with the FORTRAN resources
FORTRAN_DIR = ROOT_DIR + '/.bld/fortran'

HUGE_FLOAT = 1.0e20
SMALL_FLOAT = 1e-5

# Interpolation
INADMISSIBILITY_PENALTY = -40000.00

# Missing values. These allow to aline the treatment of missing values across
# implementations. There is no NAN available in FORTRAN.
MISSING_INT = -99
MISSING_FLOAT = -99.00

# Flag that indicate whether the parallel executables are available.
IS_PARALLEL = os.path.exists(FORTRAN_DIR + '/resfort_parallel_master')
