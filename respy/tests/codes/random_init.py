""" This module contains the functions for the generation of random requests.
"""
import numpy as np

from respy.python.shared.shared_auxiliary import generate_optimizer_options
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import IS_PARALLEL
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import NUM_PARAS

from codes.process_constraints import process_constraints
from codes.auxiliary import get_valid_values
from codes.auxiliary import get_valid_bounds
from codes.auxiliary import OPTIMIZERS_EST
from codes.auxiliary import OPTIMIZERS_AMB

MAX_AGENTS = 1000
MAX_DRAWS = 100
MAX_PERIODS = 5


def generate_init(constr=None):
    """ Get a random initialization file.
    """
    # Antibugging. This interface is using a sentinel value.
    if constr is not None:
        assert (isinstance(constr, dict))

    dict_ = generate_random_dict(constr)

    print_init_dict(dict_)

    # Finishing.
    return dict_


def generate_random_dict(constr=None):
    """ Draw random dictionary instance that can be processed into an
        initialization file.
    """
    # Antibugging. This interface is using a sentinal value.
    if constr is not None:
        assert (isinstance(constr, dict))
    else:
        constr = dict()

    # Initialize container
    dict_ = dict()

    # We now draw all parameter values. This is necessarily done here as we
    # subsequently determine a set of valid bounds.
    paras_values = []
    for i in range(NUM_PARAS):
        if i in [0]:
            value = get_valid_values('delta')
        elif i in [1]:
            value = get_valid_values('amb')
        elif i in range(2, 25):
            value = get_valid_values('coeff')
        elif i in [25, 29, 32, 34]:
            value = get_valid_values('cov')
        else:
            value = 0.0

        paras_values += [value]

    # Construct a set of valid bounds. Note that there are now bounds for the
    # coefficients of the covariance matrix. It is not clear how to enforce
    # these during an estimation on the Cholesky factors. Same problem occurs
    # for the set of fixed parameters.
    paras_bounds = []
    for i, value in enumerate(paras_values):
        if i in [0]:
            bounds = get_valid_bounds('delta', value)
        elif i in [1]:
            bounds = get_valid_bounds('amb', value)
        elif i in range(25, NUM_PARAS):
            bounds = get_valid_bounds('cov', value)
        else:
            bounds = get_valid_bounds('coeff', value)

        paras_bounds += [bounds]

    # The dictionary also contains the information whether parameters are
    # fixed during an estimation. We need to ensure that at least one
    # parameter is always free. At this point we also want to ensure that
    # either all shock coefficients are fixed or none. It is not clear how to
    # ensure other constraints on the Cholesky factors.
    paras_fixed = np.random.choice([True, False], 25).tolist()
    if sum(paras_fixed) == 25:
        paras_fixed[np.random.randint(0, 25)] = True
    paras_fixed += [np.random.choice([True, False]).tolist()] * 10

    # Sampling number of agents for the simulation. This is then used as the
    # upper bound for the dataset used in the estimation.
    num_agents_sim = np.random.randint(3, MAX_AGENTS)

    # Basics
    dict_['BASICS'] = dict()
    lower, upper = 0, 1
    dict_['BASICS']['periods'] = np.random.randint(1, MAX_PERIODS)
    dict_['BASICS']['coeffs'] = paras_values[lower:upper]
    dict_['BASICS']['bounds'] = paras_bounds[lower:upper]
    dict_['BASICS']['fixed'] = paras_fixed[lower:upper]

    # Occupation A
    lower, upper = 2, 11
    dict_['OCCUPATION A'] = dict()
    dict_['OCCUPATION A']['coeffs'] = paras_values[lower:upper]
    dict_['OCCUPATION A']['bounds'] = paras_bounds[lower:upper]
    dict_['OCCUPATION A']['fixed'] = paras_fixed[lower:upper]

    # Occupation B
    lower, upper = 11, 20
    dict_['OCCUPATION B'] = dict()
    dict_['OCCUPATION B']['coeffs'] = paras_values[lower:upper]
    dict_['OCCUPATION B']['bounds'] = paras_bounds[lower:upper]
    dict_['OCCUPATION B']['fixed'] = paras_fixed[lower:upper]

    # Education
    lower, upper = 20, 24
    dict_['EDUCATION'] = dict()
    dict_['EDUCATION']['coeffs'] = paras_values[lower:upper]
    dict_['EDUCATION']['bounds'] = paras_bounds[lower:upper]
    dict_['EDUCATION']['fixed'] = paras_fixed[lower:upper]

    dict_['EDUCATION']['start'] = np.random.randint(1, 10)
    dict_['EDUCATION']['max'] = np.random.randint(
        dict_['EDUCATION']['start'] + 1, 20)

    # Home
    lower, upper = 24, 25
    dict_['HOME'] = dict()
    dict_['HOME']['coeffs'] = paras_values[lower:upper]
    dict_['HOME']['bounds'] = paras_bounds[lower:upper]
    dict_['HOME']['fixed'] = paras_fixed[lower:upper]

    # SOLUTION
    dict_['SOLUTION'] = dict()
    dict_['SOLUTION']['draws'] = np.random.randint(1, MAX_DRAWS)
    dict_['SOLUTION']['seed'] = np.random.randint(1, 10000)
    dict_['SOLUTION']['store'] = np.random.choice(['True', 'False'])

    # AMBIGUITY
    dict_['AMBIGUITY'] = dict()
    dict_['AMBIGUITY']['mean'] = np.random.choice(['True', 'False'])
    dict_['AMBIGUITY']['measure'] = np.random.choice(['abs', 'kl'])
    dict_['AMBIGUITY']['coeffs'] = paras_values[1:2]
    dict_['AMBIGUITY']['bounds'] = paras_bounds[1:2]
    dict_['AMBIGUITY']['fixed'] = paras_fixed[1:2]

    # ESTIMATION
    dict_['ESTIMATION'] = dict()
    dict_['ESTIMATION']['agents'] = np.random.randint(1, num_agents_sim)
    dict_['ESTIMATION']['draws'] = np.random.randint(1, MAX_DRAWS)
    dict_['ESTIMATION']['seed'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['file'] = 'data.respy.dat'
    dict_['ESTIMATION']['optimizer'] = np.random.choice(OPTIMIZERS_EST)
    dict_['ESTIMATION']['maxfun'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['tau'] = np.random.uniform(100, 500)

    # DERIVATIVES
    dict_['DERIVATIVES'] = dict()
    dict_['DERIVATIVES']['version'] = 'FORWARD-DIFFERENCES'

    # PRECONDITIONING
    dict_['PRECONDITIONING'] = dict()
    dict_['PRECONDITIONING']['minimum'] = np.random.uniform(0.0000001, 0.1)
    dict_['PRECONDITIONING']['type'] = np.random.choice(['gradient',
                                                         'identity', 'magnitudes'])
    dict_['PRECONDITIONING']['eps'] = np.random.uniform(0.0000001, 0.1)

    # PROGRAM
    dict_['PROGRAM'] = dict()
    if IS_PARALLEL:
        dict_['PROGRAM']['procs'] = np.random.randint(1, 5)
    else:
        dict_['PROGRAM']['procs'] = 1

    versions = ['FORTRAN', 'PYTHON']
    if dict_['PROGRAM']['procs'] > 1:
        versions = ['FORTRAN']

    if not IS_FORTRAN:
        versions = ['PYTHON']

    dict_['PROGRAM']['debug'] = 'True'
    dict_['PROGRAM']['version'] = np.random.choice(versions)

    # The optimizer has to align with the Program version.
    if dict_['PROGRAM']['version'] == 'FORTRAN':
        dict_['ESTIMATION']['optimizer'] = np.random.choice(OPT_EST_FORT)
    else:
        dict_['ESTIMATION']['optimizer'] = np.random.choice(OPT_EST_PYTH)

    # SIMULATION
    dict_['SIMULATION'] = dict()
    dict_['SIMULATION']['seed'] = np.random.randint(1, 10000)
    dict_['SIMULATION']['agents'] = num_agents_sim
    dict_['SIMULATION']['file'] = 'data'

    # SHOCKS
    lower, upper = 25, NUM_PARAS
    dict_['SHOCKS'] = dict()
    dict_['SHOCKS']['coeffs'] = paras_values[lower:upper]
    dict_['SHOCKS']['bounds'] = paras_bounds[lower:upper]
    dict_['SHOCKS']['fixed'] = paras_fixed[lower:upper]

    # INTERPOLATION
    dict_['INTERPOLATION'] = dict()
    dict_['INTERPOLATION']['flag'] = np.random.choice(['True', 'False'])
    dict_['INTERPOLATION']['points'] = np.random.randint(10, 100)

    mock = dict()
    mock['paras_fixed'] = paras_fixed
    for optimizer in OPTIMIZERS_EST + OPTIMIZERS_AMB:
        dict_[optimizer] = generate_optimizer_options(optimizer, mock)

    # The options for the optimizers across the program versions are
    # identical. Otherwise it is not possible to simply run the solution of a
    # model with just changing the program version.
    dict_['FORT-SLSQP'] = dict_['SCIPY-SLSQP']

    # We now impose selected constraints on the final model specification.
    # These constraints can be very useful in the generation of test cases.
    dict_ = process_constraints(dict_, constr, paras_fixed, paras_bounds)

    # Finishing
    return dict_


