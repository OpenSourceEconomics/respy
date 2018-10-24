""" This module contains the functions for the generation of random requests.
"""
import numpy as np

from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_auxiliary import get_valid_bounds
from respy.pre_processing.model_processing import write_init_file
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import IS_FORTRAN

from respy.tests.codes.process_constraints import process_constraints
from respy.tests.codes.auxiliary import get_valid_shares
from respy.tests.codes.auxiliary import get_valid_values
from respy.tests.codes.auxiliary import OPTIMIZERS_EST

# We need to impose some version-dependent constraints. Otherwise the execution times for some
# tasks just takes too long.
VERSION_CONSTRAINTS = dict()

VERSION_CONSTRAINTS["max_types"] = dict()
VERSION_CONSTRAINTS["max_types"]["FORTRAN"] = 4
VERSION_CONSTRAINTS["max_types"]["PYTHON"] = 3

VERSION_CONSTRAINTS["max_periods"] = dict()
VERSION_CONSTRAINTS["max_periods"]["FORTRAN"] = 10
VERSION_CONSTRAINTS["max_periods"]["PYTHON"] = 3

VERSION_CONSTRAINTS["max_edu_start"] = dict()
VERSION_CONSTRAINTS["max_edu_start"]["FORTRAN"] = 4
VERSION_CONSTRAINTS["max_edu_start"]["PYTHON"] = 3


def generate_init(constr=None):
    """ Get a random initialization file.
    """
    # Antibugging. This interface is using a sentinel value.
    if constr is not None:
        assert isinstance(constr, dict)

    dict_ = generate_random_dict(constr)

    write_init_file(dict_)

    # Finishing.
    return dict_


def generate_random_dict(constr=None):
    """ Draw random dictionary instance that can be processed into an
        initialization file.
    """
    # Antibugging. This interface is using a sentinal value.
    if constr is not None:
        assert isinstance(constr, dict)
    else:
        constr = dict()

    # Initialize container
    dict_ = dict()

    if "version" in constr.keys():
        version = constr["version"]
    elif not IS_FORTRAN:
        version = "PYTHON"
    else:
        version = np.random.choice(["FORTRAN", "PYTHON"])

    max_edu_start = VERSION_CONSTRAINTS["max_edu_start"][version]
    max_periods = VERSION_CONSTRAINTS["max_periods"][version]
    max_types = VERSION_CONSTRAINTS["max_types"][version]
    max_agents = 1000
    max_draws = 100

    # We need to determine the final number of types right here, as it determines the number of
    # parameters. This includes imposing constraints.
    num_types = np.random.choice(range(1, max_types))
    if "types" in constr.keys():
        # Extract objects
        num_types = constr["types"]
        # Checks
        assert isinstance(num_types, int)
        assert num_types > 0

    num_paras = 54 + (num_types - 1) * 6

    # We now draw all parameter values. This is necessarily done here as we subsequently
    # determine a set of valid bounds.
    paras_values = []
    for i in range(num_paras):
        if i in [0]:
            value = get_valid_values("delta")
        elif i in range(1, 43):
            value = get_valid_values("coeff")
        elif i in [43, 47, 50, 52]:
            value = get_valid_values("cov")
        elif i in range(53, 53 + num_paras):
            value = get_valid_values("coeff")
        else:
            value = 0.0

        paras_values += [value]

    # Construct a set of valid bounds. Note that there are now bounds for the coefficients of the
    #  covariance matrix. It is not clear how to enforce these during an estimation on the
    # Cholesky factors. Same problem occurs for the set of fixed parameters.
    paras_bounds = []
    for i, value in enumerate(paras_values):
        if i in [0]:
            bounds = get_valid_bounds("delta", value)
        elif i in range(43, 53):
            bounds = get_valid_bounds("cov", value)
        else:
            bounds = get_valid_bounds("coeff", value)

        paras_bounds += [bounds]

    # The dictionary also contains the information whether parameters are fixed during an
    # estimation. We need to ensure that at least one parameter is always free. At this point we
    # also want to ensure that either all shock coefficients are fixed or none. It is not clear
    # how to ensure other constraints on the Cholesky factors.
    paras_fixed = np.random.choice([True, False], 43).tolist()
    if sum(paras_fixed) == 43:
        paras_fixed[np.random.randint(0, 43)] = True
    paras_fixed += [np.random.choice([True, False]).tolist()] * 10
    paras_fixed += np.random.choice([True, False], (num_types - 1) * 6).tolist()

    # Sampling number of agents for the simulation. This is then used as the upper bound for the
    # dataset used in the estimation.
    num_agents_sim = np.random.randint(3, max_agents)

    # Basics
    dict_["BASICS"] = dict()
    lower, upper = 0, 1
    dict_["BASICS"]["periods"] = np.random.randint(1, max_periods)
    dict_["BASICS"]["coeffs"] = paras_values[lower:upper]
    dict_["BASICS"]["bounds"] = paras_bounds[lower:upper]
    dict_["BASICS"]["fixed"] = paras_fixed[lower:upper]

    # Common Returns
    lower, upper = 1, 3
    dict_["COMMON"] = dict()
    dict_["COMMON"]["coeffs"] = paras_values[lower:upper]
    dict_["COMMON"]["bounds"] = paras_bounds[lower:upper]
    dict_["COMMON"]["fixed"] = paras_fixed[lower:upper]
    # Occupation A
    lower, upper = 3, 18
    dict_["OCCUPATION A"] = dict()
    dict_["OCCUPATION A"]["coeffs"] = paras_values[lower:upper]
    dict_["OCCUPATION A"]["bounds"] = paras_bounds[lower:upper]
    dict_["OCCUPATION A"]["fixed"] = paras_fixed[lower:upper]

    # Occupation B
    lower, upper = 18, 33
    dict_["OCCUPATION B"] = dict()
    dict_["OCCUPATION B"]["coeffs"] = paras_values[lower:upper]
    dict_["OCCUPATION B"]["bounds"] = paras_bounds[lower:upper]
    dict_["OCCUPATION B"]["fixed"] = paras_fixed[lower:upper]

    # Education
    lower, upper = 33, 40
    dict_["EDUCATION"] = dict()
    dict_["EDUCATION"]["coeffs"] = paras_values[lower:upper]
    dict_["EDUCATION"]["bounds"] = paras_bounds[lower:upper]
    dict_["EDUCATION"]["fixed"] = paras_fixed[lower:upper]

    num_edu_start = np.random.choice(range(1, max_edu_start))
    dict_["EDUCATION"]["start"] = np.random.choice(
        range(1, 20), size=num_edu_start, replace=False
    ).tolist()
    dict_["EDUCATION"]["lagged"] = np.random.uniform(size=num_edu_start).tolist()
    dict_["EDUCATION"]["share"] = get_valid_shares(num_edu_start)
    dict_["EDUCATION"]["max"] = np.random.randint(
        max(dict_["EDUCATION"]["start"]) + 1, 30
    )

    # Home
    lower, upper = 40, 43
    dict_["HOME"] = dict()
    dict_["HOME"]["coeffs"] = paras_values[lower:upper]
    dict_["HOME"]["bounds"] = paras_bounds[lower:upper]
    dict_["HOME"]["fixed"] = paras_fixed[lower:upper]

    # SOLUTION
    dict_["SOLUTION"] = dict()
    dict_["SOLUTION"]["draws"] = np.random.randint(1, max_draws)
    dict_["SOLUTION"]["seed"] = np.random.randint(1, 10000)
    dict_["SOLUTION"]["store"] = np.random.choice(["True", "False"])

    # ESTIMATION
    dict_["ESTIMATION"] = dict()
    dict_["ESTIMATION"]["agents"] = np.random.randint(1, num_agents_sim)
    dict_["ESTIMATION"]["draws"] = np.random.randint(1, max_draws)
    dict_["ESTIMATION"]["seed"] = np.random.randint(1, 10000)
    dict_["ESTIMATION"]["file"] = "data.respy.dat"
    dict_["ESTIMATION"]["optimizer"] = np.random.choice(OPTIMIZERS_EST)
    dict_["ESTIMATION"]["maxfun"] = np.random.randint(1, 10000)
    dict_["ESTIMATION"]["tau"] = np.random.uniform(100, 500)

    # DERIVATIVES
    dict_["DERIVATIVES"] = dict()
    dict_["DERIVATIVES"]["version"] = "FORWARD-DIFFERENCES"

    # PRECONDITIONING
    dict_["PRECONDITIONING"] = dict()
    dict_["PRECONDITIONING"]["minimum"] = np.random.uniform(0.0000001, 0.001)
    dict_["PRECONDITIONING"]["type"] = np.random.choice(
        ["gradient", "identity", "magnitudes"]
    )
    dict_["PRECONDITIONING"]["eps"] = np.random.uniform(0.0000001, 0.1)

    # PROGRAM
    dict_["PROGRAM"] = dict()
    dict_["PROGRAM"]["version"] = version
    dict_["PROGRAM"]["debug"] = True
    dict_["PROGRAM"]["threads"] = 1
    dict_["PROGRAM"]["procs"] = 1

    if version == "FORTRAN":
        if IS_PARALLELISM_MPI:
            dict_["PROGRAM"]["procs"] = np.random.randint(1, 5)
        if IS_PARALLELISM_OMP:
            dict_["PROGRAM"]["threads"] = np.random.randint(1, 5)

    # The optimizer has to align with the Program version.
    if dict_["PROGRAM"]["version"] == "FORTRAN":
        dict_["ESTIMATION"]["optimizer"] = np.random.choice(OPT_EST_FORT)
    else:
        dict_["ESTIMATION"]["optimizer"] = np.random.choice(OPT_EST_PYTH)

    # SIMULATION
    dict_["SIMULATION"] = dict()
    dict_["SIMULATION"]["seed"] = np.random.randint(1, 10000)
    dict_["SIMULATION"]["agents"] = num_agents_sim
    dict_["SIMULATION"]["file"] = "data"

    # SHOCKS
    lower, upper = 43, 53
    dict_["SHOCKS"] = dict()
    dict_["SHOCKS"]["coeffs"] = paras_values[lower:upper]
    dict_["SHOCKS"]["bounds"] = paras_bounds[lower:upper]
    dict_["SHOCKS"]["fixed"] = paras_fixed[lower:upper]

    lower, upper = 53, 53 + (num_types - 1) * 2
    dict_["TYPE SHARES"] = dict()
    dict_["TYPE SHARES"]["coeffs"] = paras_values[lower:upper]
    dict_["TYPE SHARES"]["bounds"] = paras_bounds[lower:upper]
    dict_["TYPE SHARES"]["fixed"] = paras_fixed[lower:upper]

    lower, upper = 53 + (num_types - 1) * 2, num_paras
    dict_["TYPE SHIFTS"] = dict()
    dict_["TYPE SHIFTS"]["coeffs"] = paras_values[lower:upper]
    dict_["TYPE SHIFTS"]["bounds"] = paras_bounds[lower:upper]
    dict_["TYPE SHIFTS"]["fixed"] = paras_fixed[lower:upper]

    # INTERPOLATION
    dict_["INTERPOLATION"] = dict()
    dict_["INTERPOLATION"]["flag"] = np.random.choice(["True", "False"])
    dict_["INTERPOLATION"]["points"] = np.random.randint(10, 100)

    mock = dict()
    mock["paras_fixed"] = paras_fixed
    for optimizer in OPTIMIZERS_EST:
        dict_[optimizer] = generate_optimizer_options(optimizer, mock, num_paras)

    # We now impose selected constraints on the final model specification. These constraints can
    # be very useful in the generation of test cases.
    dict_ = process_constraints(dict_, constr, paras_fixed, paras_bounds)

    # Finishing
    return dict_


def generate_optimizer_options(which, optim_paras, num_paras):

    dict_ = dict()

    if which == "SCIPY-BFGS":
        dict_["gtol"] = np.random.uniform(0.0000001, 0.1)
        dict_["maxiter"] = np.random.randint(1, 10)
        dict_["eps"] = np.random.uniform(1e-9, 1e-6)

    elif which == "SCIPY-LBFGSB":
        dict_["factr"] = np.random.uniform(10, 100)
        dict_["pgtol"] = np.random.uniform(1e-6, 1e-4)
        dict_["maxiter"] = np.random.randint(1, 10)
        dict_["maxls"] = np.random.randint(1, 10)
        dict_["m"] = np.random.randint(1, 10)
        dict_["eps"] = np.random.uniform(1e-9, 1e-6)

    elif which == "SCIPY-POWELL":
        dict_["xtol"] = np.random.uniform(0.0000001, 0.1)
        dict_["ftol"] = np.random.uniform(0.0000001, 0.1)
        dict_["maxfun"] = np.random.randint(1, 100)
        dict_["maxiter"] = np.random.randint(1, 100)

    elif which in ["FORT-NEWUOA", "FORT-BOBYQA"]:
        rhobeg = np.random.uniform(0.0000001, 0.001)
        dict_["maxfun"] = np.random.randint(1, 100)
        dict_["rhobeg"] = rhobeg
        dict_["rhoend"] = np.random.uniform(0.01, 0.99) * rhobeg

        # It is not recommended that N is larger than upper as the code might
        # break down due to a segmentation fault. See the source files for the
        # absolute upper bounds.
        assert sum(optim_paras["paras_fixed"]) != num_paras
        lower = (num_paras - sum(optim_paras["paras_fixed"])) + 2
        upper = 2 * (num_paras - sum(optim_paras["paras_fixed"])) + 1
        dict_["npt"] = np.random.randint(lower, upper + 1)

    elif which == "FORT-BFGS":
        dict_["maxiter"] = np.random.randint(1, 100)
        dict_["stpmx"] = np.random.uniform(75, 125)
        dict_["gtol"] = np.random.uniform(0.0001, 0.1)
        dict_["eps"] = np.random.uniform(1e-9, 1e-6)

    else:
        raise NotImplementedError("The optimizer you requested is not implemented.")

    return dict_
