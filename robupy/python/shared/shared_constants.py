""" Module for program constants used across the ROBUPY package. This is
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
FORTRAN_DIR = ROOT_DIR + '/fortran'

HUGE_FLOAT = 1.0e10
TINY_FLOAT = 1e-20
SMALL_FLOAT = 1e-5

# Interpolation
INTERPOLATION_INADMISSIBLE_STATES = -50000.00

# Missing values. These allow to aline the treatment of missing values across
# implementations. There is no NAN available in FORTRAN.
MISSING_INT = -99
MISSING_FLOAT = float(MISSING_INT)