""" Module for program constants used across the ROBUPY package. This is
aligned with the constants from the FORTRAN implementation.
"""

# standard library
import os

# Obtain the root directory of the package
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/constants', '')

# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR + '/tests'
TEST_RESOURCES_DIR = ROOT_DIR + '/tests/resources'

# Directory with the FORTRAN resources
FORTRAN_DIR = ROOT_DIR + '/fortran'


# module-wide variables
DEBUG_OPTIONS = ' -O2 -fimplicit-none  -Wall -Wline-truncation' \
                ' -Wcharacter-truncation  -Wsurprising  -Waliasing' \
                ' -Wimplicit-interface  -Wunused-parameter  -fwhole-file' \
                ' -fcheck=all  -fbacktrace '

PRODUCTION_OPTIONS = '-O3'



MISSING_INT = -99

MISSING_FLOAT = -99.00
HUGE_FLOAT = 1.0e10
TINY_FLOAT = 1e-20
SMALL_FLOAT = 1e-5

# Interpolation
INTERPOLATION_INADMISSIBLE_STATES = -50000.00