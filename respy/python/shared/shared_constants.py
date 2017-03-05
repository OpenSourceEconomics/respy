""" Module for program constants used across the RESPY package. This is
aligned with the constants from the FORTRAN implementation.
"""
import numpy as np
import json
import os

# Obtain the root directory of the package
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/python/shared', '')

# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR + '/tests'
TEST_RESOURCES_DIR = ROOT_DIR + '/tests/resources'

# Directory with the FORTRAN resources
EXEC_DIR = ROOT_DIR + '/fortran/bin'

MINISCULE_FLOAT = 1.0e-100
LARGE_FLOAT = 1.0e8
HUGE_FLOAT = 1.0e20
SMALL_FLOAT = 1e-5
TINY_FLOAT = 1.0e-8
PRINT_FLOAT = 1e10

# Ambiguity
MIN_AMBIGUITY = 1e-20

# Interpolation
INADMISSIBILITY_PENALTY = -40000.00

# Missing values. These allow to aline the treatment of missing values across
# implementations. There is no NAN available in FORTRAN.
MISSING_INT = -99
MISSING_FLOAT = -99.00

# Flags that provide additional information about the exact configuration
with open(ROOT_DIR + '/.config', 'r') as infile:
    config_dict = json.load(infile)

IS_TESTING = config_dict['TESTING']
IS_DEBUG = config_dict['DEBUG']

IS_PARALLEL = config_dict['PARALLELISM']
IS_FORTRAN = config_dict['FORTRAN']
IS_F2PY = config_dict['F2PY']

# Each implementation has its own set of optimizers available.
OPT_EST_PYTH = ['SCIPY-BFGS', 'SCIPY-POWELL', 'SCIPY-LBFGSB']
OPT_EST_FORT = ['FORT-NEWUOA', 'FORT-BFGS', 'FORT-BOBYQA']

OPT_AMB_PYTH = ['SCIPY-SLSQP']
OPT_AMB_FORT = ['FORT-SLSQP']

# Labels for columns in a dataset as well as the formatters.
LABELS = []
LABELS += ['Identifier', 'Period', 'Choice', 'Wage']
LABELS += ['Experience A', 'Experience B', 'Years Schooling']
LABELS += ['Lagged Schooling']

FORMATS_DICT = dict()
FORMATS_DICT.update({0: np.int, 1: np.int, 2: np.int, 3: np.float})
FORMATS_DICT.update({4: np.int, 5: np.int, 6: np.int, 7: np.int})
