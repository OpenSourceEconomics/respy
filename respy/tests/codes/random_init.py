""" This module contains the functions for the generation of random requests.
"""
import numpy as np

from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import IS_PARALLEL
from respy.python.shared.shared_constants import IS_FORTRAN

# module-wide variables
MAX_AGENTS = 1000
MAX_DRAWS = 100
MAX_PERIODS = 5

OPTIMIZERS = ['SCIPY-BFGS', 'SCIPY-POWELL', 'FORT-NEWUOA', 'FORT-BFGS']


def generate_init(constraints=None):
    """ Get a random initialization file.
    """
    # Antibugging. This interface is using a sentinel value.
    if constraints is not None:
        assert (isinstance(constraints, dict))

    dict_ = generate_random_dict(constraints)

    print_init_dict(dict_)

    # Finishing.
    return dict_


def generate_random_dict(constraints=None):
    """ Draw random dictionary instance that can be processed into an
        initialization file.
    """
    # Antibugging. This interface is using a sentinal value.
    if constraints is not None:
        assert (isinstance(constraints, dict))
    else:
        constraints = dict()

    # Initialize container
    dict_ = dict()

    # The dictionary also contains the information whether parameters are
    # fixed during an estimation. We need to ensure that at least one
    # parameter is always free.
    paras_fixed = np.random.choice([True, False], 26).tolist()
    if sum(paras_fixed) == 26:
        paras_fixed[np.random.randint(0, 26)] = True

    # Sampling number of agents for the simulation. This is then used as the
    # upper bound for the dataset used in the estimation.
    num_agents_sim = np.random.randint(3, MAX_AGENTS)

    # Basics
    dict_['BASICS'] = dict()
    dict_['BASICS']['periods'] = np.random.randint(1, MAX_PERIODS)
    dict_['BASICS']['delta'] = np.random.random()

    # Home
    dict_['HOME'] = dict()
    dict_['HOME']['coeffs'] = np.random.uniform(-0.05, 0.05, 1).tolist()
    dict_['HOME']['fixed'] = paras_fixed[15:16]

    # Occupation A
    dict_['OCCUPATION A'] = dict()
    dict_['OCCUPATION A']['coeffs'] = np.random.uniform(-0.05, 0.05, 6).tolist()
    dict_['OCCUPATION A']['fixed'] = paras_fixed[0:6]

    # Occupation B
    dict_['OCCUPATION B'] = dict()
    dict_['OCCUPATION B']['coeffs'] = np.random.uniform(-0.05, 0.05, 6).tolist()
    dict_['OCCUPATION B']['fixed'] = paras_fixed[6:12]

    # Education
    dict_['EDUCATION'] = dict()
    dict_['EDUCATION']['coeffs'] = np.random.uniform(-0.05, 0.05, 6).tolist()
    dict_['EDUCATION']['fixed'] = paras_fixed[12:15]

    dict_['EDUCATION']['start'] = np.random.randint(1, 10)
    dict_['EDUCATION']['max'] = np.random.randint(
        dict_['EDUCATION']['start'] + 1, 20)

    # SOLUTION
    dict_['SOLUTION'] = dict()
    dict_['SOLUTION']['draws'] = np.random.randint(1, MAX_DRAWS)
    dict_['SOLUTION']['seed'] = np.random.randint(1, 10000)
    dict_['SOLUTION']['store'] = np.random.choice(['True', 'False'])

    # AMBIGUITY
    dict_['AMBIGUITY'] = dict()
    dict_['AMBIGUITY']['measure'] = np.random.choice(['abs', 'kl'])
    dict_['AMBIGUITY']['level'] = np.random.choice([0.0, np.random.uniform()])

    # ESTIMATION
    dict_['ESTIMATION'] = dict()
    dict_['ESTIMATION']['agents'] = np.random.randint(1, num_agents_sim)
    dict_['ESTIMATION']['draws'] = np.random.randint(1, MAX_DRAWS)
    dict_['ESTIMATION']['seed'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['file'] = 'data.respy.dat'
    dict_['ESTIMATION']['optimizer'] = np.random.choice(OPTIMIZERS)
    dict_['ESTIMATION']['maxfun'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['tau'] = np.random.uniform(100, 500)

    # DERIVATIVES
    dict_['DERIVATIVES'] = dict()
    dict_['DERIVATIVES']['version'] = 'FORWARD-DIFFERENCES'
    dict_['DERIVATIVES']['eps'] = np.random.uniform(0.0000001, 0.1)

    # SCALING
    dict_['SCALING'] = dict()
    dict_['SCALING']['minimum'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCALING']['flag'] = np.random.choice([True, False])

    # PARALLELISM
    dict_['PARALLELISM'] = dict()
    dict_['PARALLELISM']['procs'] = np.random.randint(2, 5)

    # Parallelism is only supported in FORTRAN implementation.
    if IS_PARALLEL:
        dict_['PARALLELISM']['flag'] = np.random.choice([True, False])
    else:
        dict_['PARALLELISM']['flag'] = False

    versions = ['FORTRAN', 'PYTHON']
    if dict_['PARALLELISM']['flag']:
        versions = ['FORTRAN']

    if not IS_FORTRAN:
        versions = ['PYTHON']

    # PROGRAM
    dict_['PROGRAM'] = dict()
    dict_['PROGRAM']['debug'] = 'True'

    dict_['PROGRAM']['version'] = np.random.choice(versions)

    # The optimizer has to align with the Program version.
    if dict_['PROGRAM']['version'] == 'FORTRAN':
        dict_['ESTIMATION']['optimizer'] = np.random.choice(['FORT-NEWUOA',
            'FORT-BFGS'])
    else:
        dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS',
            'SCIPY-POWELL'])

    # SIMULATION
    dict_['SIMULATION'] = dict()
    dict_['SIMULATION']['seed'] = np.random.randint(1, 10000)
    dict_['SIMULATION']['agents'] = num_agents_sim
    dict_['SIMULATION']['file'] = 'data.respy.dat'

    # SHOCKS
    dict_['SHOCKS'] = dict()
    shocks = np.zeros(10)
    for i in [0, 4, 7, 9]:
        shocks[i] = np.random.uniform(0.05, 1)
    dict_['SHOCKS']['coeffs'] = shocks
    dict_['SHOCKS']['fixed'] = np.array(paras_fixed[16:])

    # INTERPOLATION
    dict_['INTERPOLATION'] = dict()
    dict_['INTERPOLATION']['flag'] = np.random.choice([True, False])
    dict_['INTERPOLATION']['points'] = np.random.randint(10, 100)

    # SCIPY-BFGS
    dict_['SCIPY-BFGS'] = dict()
    dict_['SCIPY-BFGS']['gtol'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCIPY-BFGS']['maxiter'] = np.random.randint(1, 10)

    # SCIPY-BFGS
    dict_['SCIPY-POWELL'] = dict()
    dict_['SCIPY-POWELL']['xtol'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCIPY-POWELL']['ftol'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCIPY-POWELL']['maxfun'] = np.random.randint(1, 100)
    dict_['SCIPY-POWELL']['maxiter'] = np.random.randint(1, 100)

    # FORT-NEWUOA
    rhobeg = np.random.uniform(0.0000001, 0.1)

    dict_['FORT-NEWUOA'] = dict()
    dict_['FORT-NEWUOA']['maxfun'] = np.random.randint(1, 100)
    dict_['FORT-NEWUOA']['rhobeg'] = rhobeg
    dict_['FORT-NEWUOA']['rhoend'] = np.random.uniform(0.01, 0.99) * rhobeg

    lower = (26 - sum(paras_fixed)) + 2
    upper = (2 * (26 - sum(paras_fixed)) + 1)
    dict_['FORT-NEWUOA']['npt'] = np.random.randint(lower, upper)

    # FORT-BFGS
    dict_['FORT-BFGS'] = dict()
    dict_['FORT-BFGS']['maxiter'] = np.random.randint(1, 100)
    dict_['FORT-BFGS']['stpmx'] = np.random.uniform(75, 125)
    dict_['FORT-BFGS']['gtol'] = np.random.uniform(0.0001, 0.1)

    """ We now impose selected constraints on the final model specification.
    These constraints can be very useful in the generation of test cases.
    """

    # Address incompatibility issues
    keys = constraints.keys()
    if 'is_myopic' in keys:
        assert 'delta' not in keys

    if 'is_estimation' in keys:
        assert 'maxfun' not in keys
        assert 'flag_scaling' not in keys

    if 'agents' in keys:
        assert 'max_draws' not in keys

    if ('flag_parallelism' in keys) and ('version' in keys) and constraints[
        'flag_parallelism']:
            assert constraints['version'] == 'FORTRAN'

    # Replace interpolation
    if 'flag_interpolation' in constraints.keys():
        # Checks
        assert (constraints['flag_interpolation'] in [True, False])
        # Replace in initialization files
        dict_['INTERPOLATION']['flag'] = constraints['flag_interpolation']

    # Replace number of periods
    if 'points' in constraints.keys():
        # Extract objects
        points = constraints['points']
        # Checks
        assert (isinstance(points, int))
        assert (points > 0)
        # Replace in initialization files
        dict_['INTERPOLATION']['points'] = points

    # Replace number of iterations
    if 'maxfun' in constraints.keys():
        # Extract objects
        maxfun = constraints['maxfun']
        # Checks
        assert (isinstance(maxfun, int))
        assert (maxfun >= 0)
        # Replace in initialization files
        dict_['ESTIMATION']['maxfun'] = maxfun

    # Replace education
    if 'edu' in constraints.keys():
        # Extract objects
        start, max_ = constraints['edu']
        # Checks
        assert (isinstance(start, int))
        assert (start > 0)
        assert (isinstance(max_, int))
        assert (max_ > start)
        # Replace in initialization file
        dict_['EDUCATION']['start'] = start
        dict_['EDUCATION']['max'] = max_

    # Replace presence of ambiguity
    if 'is_ambiguity' in constraints.keys():
        # Extract objects
        is_ambiguity = constraints['is_ambiguity']
        # Checks
        assert (is_ambiguity in [True, False])
        # Replace initialization file
        dict_['AMBIGUITY']['flag'] = is_ambiguity

    # Replace level of ambiguity
    if 'level' in constraints.keys():
        # Extract object
        level = constraints['level']
        # Checks
        assert isinstance(level, float)
        assert level >= 0.0
        # Replace in initialization file
        dict_['AMBIGUITY']['level'] = level

    # Replace version
    if 'version' in constraints.keys():
        # Extract objects
        version = constraints['version']
        # Checks
        assert (version in ['PYTHON', 'FORTRAN'])
        # Replace in initialization file
        dict_['PROGRAM']['version'] = version
        # Ensure that the constraints are met
        if version != 'FORTRAN':
            dict_['PARALLELISM']['flag'] = False
        if version == 'FORTRAN':
            dict_['ESTIMATION']['optimizer'] = np.random.choice(['FORT-NEWUOA', 'FORT-BFGS'])
        else:
            dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])

    # Ensure that random deviates do not exceed a certain number. This is
    # useful when aligning the randomness across implementations.
    if 'max_draws' in constraints.keys():
        # Extract objects
        max_draws = constraints['max_draws']
        # Checks
        assert (isinstance(max_draws, int))
        assert (max_draws > 2)
        # Replace in initialization file
        num_agents_sim = np.random.randint(2, max_draws)
        dict_['SIMULATION']['agents'] = num_agents_sim
        dict_['ESTIMATION']['agents'] = np.random.randint(1, num_agents_sim)
        dict_['ESTIMATION']['draws'] = np.random.randint(1, max_draws)
        dict_['SOLUTION']['draws'] = np.random.randint(1, max_draws)

    # Replace parallelism ...
    if 'flag_parallelism' in constraints.keys():
        # Extract objects
        flag_parallelism = constraints['flag_parallelism']
        # Checks
        assert (flag_parallelism in [True, False])
        # Replace in initialization file
        dict_['PARALLELISM']['flag'] = flag_parallelism
        # Ensure that the constraints are met
        if dict_['PARALLELISM']['flag']:
            dict_['PROGRAM']['version'] = 'FORTRAN'

    # Replace parallelism ...
    if 'flag_scaling' in constraints.keys():
        # Extract objects
        flag_scaling = constraints['flag_scaling']
        # Checks
        assert (flag_scaling in [True, False])
        # Replace in initialization file
        dict_['SCALING']['flag'] = flag_scaling

    # Replace store attribute
    if 'is_store' in constraints.keys():
        # Extract objects
        is_store = constraints['is_store']
        # Checks
        assert (is_store in [True, False])
        # Replace in initialization file
        dict_['SOLUTION']['store'] = is_store

    # Replace number of periods
    if 'periods' in constraints.keys():
        # Extract objects
        periods = constraints['periods']
        # Checks
        assert (isinstance(periods, int))
        assert (periods > 0)
        # Replace in initialization files
        dict_['BASICS']['periods'] = periods

    # Replace discount factor
    if 'is_myopic' in constraints.keys():
        # Extract object
        assert ('delta' not in constraints.keys())
        assert (constraints['is_myopic'] in [True, False])
        # Replace in initialization files
        if constraints['is_myopic']:
            dict_['BASICS']['delta'] = 0.0
        else:
            dict_['BASICS']['delta'] = np.random.uniform(0.1, 1.0)

    # Replace discount factor. This is option is needed in addition to
    # is_myopic the code is run for very small levels of delta and compared
    # against the myopic version.
    if 'delta' in constraints.keys():
        # Extract objects
        delta = constraints['delta']
        # Checks
        assert ('is_myopic' not in constraints.keys())
        assert (np.isfinite(delta))
        assert (delta >= 0.0)
        assert (isinstance(delta, float))
        # Replace in initialization file
        dict_['BASICS']['delta'] = delta

    # No random component to rewards
    if 'is_deterministic' in constraints.keys():
        # Checks
        assert (constraints['is_deterministic'] in [True, False])
        # Replace in initialization files
        if constraints['is_deterministic']:
            dict_['SHOCKS']['coeffs'] = np.zeros(10)

    # Number of agents
    if 'agents' in constraints.keys():
        # Extract object
        num_agents = constraints['agents']
        # Checks
        assert (num_agents > 0)
        assert (isinstance(num_agents, int))
        assert (np.isfinite(num_agents))
        # Replace in initialization files
        dict_['SIMULATION']['agents'] = num_agents
        if num_agents == 1:
            dict_['ESTIMATION']['agents'] = 1
        else:
            dict_['ESTIMATION']['agents'] = np.random.randint(1, num_agents)

    # Estimation task, but very small.
    if 'is_estimation' in constraints.keys():
        # Checks
        assert (constraints['is_estimation'] in [True, False])
        # Replace in initialization files
        if constraints['is_estimation']:
            dict_['ESTIMATION']['maxfun'] = np.random.randint(1, 10)
            dict_['SCALING']['flag'] = np.random.choice([True, False], p=[0.1, 0.9])

    # Finishing
    return dict_
