import numpy as np

from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH

from codes.auxiliary import get_valid_bounds

# We maintain a list of all valid constraints and check all specified keys
# against it.
VALID_KEYS = []
VALID_KEYS += ['flag_estimation', 'flag_ambiguity', 'agents']
VALID_KEYS += ['flag_parallelism', 'version', 'file_est', 'flag_interpolation']
VALID_KEYS += ['points', 'maxfun', 'flag_deterministic', 'delta']
VALID_KEYS += ['edu', 'measure', 'level', 'fixed_ambiguity', 'flag_ambiguity']
VALID_KEYS += ['max_draws', 'flag_precond', 'periods']
VALID_KEYS += ['is_store', 'is_myopic']


def process_constraints(dict_, constraints, paras_fixed, paras_bounds):
    """ Check and process constraints.
    """

    # Check request
    _check_constraints(constraints)

    # Replace path to dataset used for estimation
    if 'file_est' in constraints.keys():
        # Checks
        assert isinstance(constraints['file_est'], str)
        # Replace in initialization files
        dict_['ESTIMATION']['file'] = constraints['file_est']

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

    # Replace measure of ambiguity
    if 'measure' in constraints.keys():
        # Extract object
        measure = constraints['measure']
        # Checks
        assert measure in ['kl', 'abs']
        # Replace in initialization file
        dict_['AMBIGUITY']['measure'] = measure

    # Replace level of ambiguity
    if 'level' in constraints.keys():
        # Extract object
        level = constraints['level']
        # Checks
        assert isinstance(level, float)
        assert level >= 0.0
        # Replace in initialization file
        dict_['AMBIGUITY']['coeffs'] = [level]
        dict_['AMBIGUITY']['bounds'] = [get_valid_bounds('amb', level)]

    # Treat level of ambiguity as fixed in an estimation
    if 'flag_ambiguity' in constraints.keys():
        # Checks
        assert (constraints['flag_ambiguity'] in [True, False])
        # Replace in initialization files
        if constraints['flag_ambiguity']:
            value = np.random.uniform(0.01, 1.0)
            dict_['AMBIGUITY']['coeffs'] = [value]
            dict_['AMBIGUITY']['bounds'] = [get_valid_bounds('amb', value)]
        else:
            dict_['AMBIGUITY']['coeffs'] = [0.00]
            dict_['AMBIGUITY']['bounds'] = [get_valid_bounds('amb', 0.00)]

    # Treat level of ambiguity as fixed in an estimation
    if 'fixed_ambiguity' in constraints.keys():
        # Checks
        assert (constraints['fixed_ambiguity'] in [True, False])
        # Replace in initialization files
        dict_['AMBIGUITY']['fixed'] = [constraints['fixed_ambiguity']]

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
            dict_['PROGRAM']['procs'] = 1
        if version == 'FORTRAN':
            dict_['ESTIMATION']['optimizer'] = np.random.choice(OPT_EST_FORT)
        else:
            dict_['ESTIMATION']['optimizer'] = np.random.choice(OPT_EST_PYTH)

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
        if flag_parallelism:
            dict_['PROGRAM']['procs'] = np.random.randint(2, 5)
        else:
            dict_['PROGRAM']['procs'] = 1
        # Ensure that the constraints are met
        if dict_['PROGRAM']['procs'] > 1:
            dict_['PROGRAM']['version'] = 'FORTRAN'

    if 'flag_precond' in constraints.keys():
        # Extract objects
        flag_precond = constraints['flag_precond']
        # Checks
        assert (flag_precond in [True, False])
        # Replace in initialization file
        if flag_precond:
            dict_['PRECONDITIONING']['type'] = 'gradient'
        else:
            dict_['PRECONDITIONING']['type'] = 'identity'

    # Replace store attribute
    if 'is_store' in constraints.keys():
        # Extract objects
        is_store = constraints['is_store']
        # Checks
        assert (is_store in [True, False])
        # Replace in initialization file
        dict_['SOLUTION']['store'] = str(is_store)

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
            dict_['BASICS']['coeffs'] = [0.0]
            dict_['BASICS']['bounds'] = [get_valid_bounds('delta', 0.00)]
        else:
            value = np.random.uniform(0.01, 1.0)
            dict_['BASICS']['coeffs'] = [value]
            dict_['BASICS']['bounds'] = [get_valid_bounds('amb', value)]

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
        dict_['BASICS']['coeffs'] = delta

    # No random component to rewards
    if 'flag_deterministic' in constraints.keys():
        # Checks
        assert (constraints['flag_deterministic'] in [True, False])
        # Replace in initialization files
        if constraints['flag_deterministic']:
            dict_['SHOCKS']['coeffs'] = [0.0] * 10

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

    # Estimation task, but very small. A host of other constraints need to be
    # honored as well.
    if 'flag_estimation' in constraints.keys():
        # Checks
        assert (constraints['flag_estimation'] in [True, False])
        # Replace in initialization files
        if constraints['flag_estimation']:
            dict_['is_store'] = False
            dict_['ESTIMATION']['maxfun'] = int(np.random.choice(range(6),
                p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
            dict_['PRECONDITIONING']['type'] = \
                np.random.choice(['gradient', 'identity'], p=[0.1, 0.9])

            # Ensure that a valid estimator is selected in the case that a
            # free parameter has bounds.
            for i in range(28):
                if paras_fixed[i]:
                    continue
                if any(item is not None for item in paras_bounds[i]):
                    if dict_['PROGRAM']['version'] == 'FORTRAN':
                        dict_['ESTIMATION']['optimizer'] = 'FORT-BOBYQA'
                    else:
                        dict_['ESTIMATION']['optimizer'] = 'SCIPY-LBFGSB'
                    break

    return dict_


def _check_constraints(constraints):
    """ Check that there are no conflicting constraints imposed.
    """
    # Check all specifie dconstraints
    for key_ in constraints.keys():
        assert key_ in VALID_KEYS

    # Address incompatibility issues
    keys = constraints.keys()

    if 'is_myopic' in keys:
        assert 'delta' not in keys

    if 'flag_estimation' in keys:
        assert 'maxfun' not in keys
        assert 'flag_precond' not in keys

    if 'flag_ambiguity' in keys:
        assert 'level' not in keys

    if 'agents' in keys:
        assert 'max_draws' not in keys

    cond = ('flag_parallelism' in keys) and ('version' in keys)
    cond = cond and constraints['flag_parallelism']
    if cond:
        assert constraints['version'] == 'FORTRAN'
