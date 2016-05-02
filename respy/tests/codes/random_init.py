""" This module contains the functions for the generation of random requests.
"""

# standard library
import numpy as np

from respy.python.shared.shared_auxiliary import print_init_dict

# module-wide variables
MAX_AGENTS = 1000
MAX_DRAWS = 100
MAX_PERIODS = 5


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
    dict_['BASICS'] = {}
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
    dict_['SOLUTION'] = {}
    dict_['SOLUTION']['draws'] = np.random.randint(1, MAX_DRAWS)
    dict_['SOLUTION']['seed'] = np.random.randint(1, 10000)
    dict_['SOLUTION']['store'] = np.random.choice(['True', 'False'])

    # ESTIMATION
    dict_['ESTIMATION'] = {}
    dict_['ESTIMATION']['agents'] = np.random.randint(1, num_agents_sim)
    dict_['ESTIMATION']['draws'] = np.random.randint(1, MAX_DRAWS)
    dict_['ESTIMATION']['seed'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['file'] = 'data.respy'
    dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])
    dict_['ESTIMATION']['maxiter'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['tau'] = np.random.uniform(100, 500)

    # PROGRAM
    dict_['PROGRAM'] = {}
    dict_['PROGRAM']['debug'] = 'True'
    dict_['PROGRAM']['version'] = np.random.choice(['FORTRAN', 'F2PY', 'PYTHON'])

    # SIMULATION
    dict_['SIMULATION'] = {}
    dict_['SIMULATION']['seed'] = np.random.randint(1, 10000)
    dict_['SIMULATION']['agents'] = num_agents_sim
    dict_['SIMULATION']['file'] = 'data.respy'

    # SHOCKS
    dict_['SHOCKS'] = dict()
    shocks = np.zeros(10)
    for i in [0, 4, 7, 9]:
        shocks[i] = np.random.uniform(0.05, 1)
    dict_['SHOCKS']['coeffs'] = shocks
    dict_['SHOCKS']['fixed'] = np.array(paras_fixed[16:])

    # INTERPOLATION
    dict_['INTERPOLATION'] = {}
    dict_['INTERPOLATION']['apply'] = np.random.choice([True, False])
    dict_['INTERPOLATION']['points'] = np.random.randint(10, 100)

    # SCIPY-BFGS
    dict_['SCIPY-BFGS'] = {}
    dict_['SCIPY-BFGS']['epsilon'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCIPY-BFGS']['gtol'] = np.random.uniform(0.0000001, 0.1)

    # SCIPY-BFGS
    dict_['SCIPY-POWELL'] = {}
    dict_['SCIPY-POWELL']['xtol'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCIPY-POWELL']['ftol'] = np.random.uniform(0.0000001, 0.1)
    dict_['SCIPY-POWELL']['maxfun'] = np.random.randint(1, 100)

    '''We now impose selected constraints on the final model specification.
       These constraints can be very useful in the generation of test cases. '''

    # Address incompatibility issues
    keys = constraints.keys()
    if 'is_myopic' in keys:
        assert 'delta' not in keys

    if 'is_estimation' in keys:
        assert 'maxiter' not in keys

    if 'agents' in keys:
        assert 'max_draws' not in keys

    # Replace interpolation
    if 'apply' in constraints.keys():
        # Checks
        assert (constraints['apply'] in [True, False])
        # Replace in initialization files
        dict_['INTERPOLATION']['apply'] = constraints['apply']

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
    if 'maxiter' in constraints.keys():
        # Extract objects
        maxiter = constraints['maxiter']
        # Checks
        assert (isinstance(maxiter, int))
        assert (maxiter >= 0)
        # Replace in initialization files
        dict_['ESTIMATION']['maxiter'] = maxiter

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

    # Replace version
    if 'version' in constraints.keys():
        # Extract objects
        version = constraints['version']
        # Checks
        assert (version in ['PYTHON', 'FORTRAN', 'F2PY'])
        # Replace in initialization file
        dict_['PROGRAM']['version'] = version

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

    # Replace store attribute
    if 'store' in constraints.keys():
        # Extract objects
        store = constraints['store']
        # Checks
        assert (store in [True, False])
        # Replace in initialization file
        dict_['SOLUTION']['store'] = store

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

    # No random component to payoffs
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
            dict_['SCIPY-POWELL']['maxfun'] = 1
            dict_['ESTIMATION']['maxiter'] = 1

    # Finishing
    return dict_


