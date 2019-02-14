""" Module for program constants used across the RESPY package. This is
aligned with the constants from the FORTRAN implementation.
"""
import numpy as np
import json
import os

# Obtain the root directory of the package
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
# ROOT_DIR = ROOT_DIR.replace("/python/shared", "")

# Directory with additional resources for the testing harness
# TEST_DIR = ROOT_DIR + "/tests"
TEST_RESOURCES_DIR = os.getcwd() + "/respy/tests/resources"
TEST_RESOURCES_BUILD = ROOT_DIR + "/.bld/tests/resources"

# Directory with the FORTRAN resources
EXEC_DIR = ROOT_DIR + "/.bld/fortran"

MINISCULE_FLOAT = 1.0e-100
LARGE_FLOAT = 1.0e8
HUGE_FLOAT = 1.0e20
SMALL_FLOAT = 1e-5
TINY_FLOAT = 1.0e-8
PRINT_FLOAT = 1e10

# Number of decimals that are compared for tests
# This is currently only used in regression tests.
DECIMALS = 6
# Some assert fucntions take rtol instead of decimals
TOL = 10 ** -DECIMALS

# Interpolation
INADMISSIBILITY_PENALTY = -400000.00

# Missing values. These allow to aline the treatment of missing values across
# implementations. There is no NAN available in FORTRAN.
MISSING_INT = -99
MISSING_FLOAT = -99.00

# TODO(janosg): Raises exception if installed on Windows without Fortran.
try:
    # Flags that provide additional information about the exact configuration
    with open(ROOT_DIR + "/.bld/.config", "r") as infile:
        config_dict = json.load(infile)
except FileNotFoundError:
    json_string = (
        '{"DEBUG": false, "FORTRAN": false, "F2PY": false, "PARALLELISM_MPI": false, '
        '"PARALLELISM_OMP": false}'
    )
    config_dict = json.loads(json_string)

IS_DEBUG = config_dict["DEBUG"]

IS_PARALLELISM_MPI = config_dict["PARALLELISM_MPI"]
IS_PARALLELISM_OMP = config_dict["PARALLELISM_OMP"]
IS_FORTRAN = config_dict["FORTRAN"]
IS_F2PY = config_dict["F2PY"]

# Each implementation has its own set of optimizers available.
OPT_EST_PYTH = ["SCIPY-BFGS", "SCIPY-POWELL", "SCIPY-LBFGSB"]
OPT_EST_FORT = ["FORT-NEWUOA", "FORT-BFGS", "FORT-BOBYQA"]

# Labels for columns in a dataset as well as the formatters.
DATA_LABELS_EST = []
DATA_LABELS_EST += ["Identifier", "Period", "Choice", "Wage"]
DATA_LABELS_EST += ["Experience_A", "Experience_B", "Years_Schooling"]
DATA_LABELS_EST += ["Lagged_Choice"]

# There is additional information available in a simulated dataset.
DATA_LABELS_SIM = DATA_LABELS_EST[:]
DATA_LABELS_SIM += ["Type"]
DATA_LABELS_SIM += ["Total_Reward_1", "Total_Reward_2"]
DATA_LABELS_SIM += ["Total_Reward_3", "Total_Reward_4"]
DATA_LABELS_SIM += ["Systematic_Reward_1", "Systematic_Reward_2"]
DATA_LABELS_SIM += ["Systematic_Reward_3", "Systematic_Reward_4"]
DATA_LABELS_SIM += ["Shock_Reward_1", "Shock_Reward_2"]
DATA_LABELS_SIM += ["Shock_Reward_3", "Shock_Reward_4"]
DATA_LABELS_SIM += [
    "Discount_Rate",
    "General_Reward_1",
    "General_Reward_2",
    "Common_Reward",
]
DATA_LABELS_SIM += ["Immediate_Reward_1", "Immediate_Reward_2", "Immediate_Reward_3"]
DATA_LABELS_SIM += ["Immediate_Reward_4"]

DATA_FORMATS_EST = dict()
for key_ in DATA_LABELS_EST:
    DATA_FORMATS_EST[key_] = np.int
    if key_ in ["Wage"]:
        DATA_FORMATS_EST[key_] = np.float

DATA_FORMATS_SIM = dict(DATA_FORMATS_EST)
for key_ in DATA_LABELS_SIM:
    if key_ in DATA_FORMATS_SIM.keys():
        continue
    elif key_ in ["Type"]:
        DATA_FORMATS_SIM[key_] = np.int
    else:
        DATA_FORMATS_SIM[key_] = np.float
