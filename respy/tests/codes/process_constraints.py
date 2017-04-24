import numpy as np

from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH

from codes.auxiliary import get_valid_bounds

# We maintain a list of all valid constraints and check all specified keys
# against it.
VALID_KEYS = []
VALID_KEYS += ['flag_estimation', 'flag_ambiguity', 'agents']
VALID_KEYS += ['flag_parallelism', 'version', 'file_est', 'flag_interpolation']
VALID_KEYS += ['points', 'maxfun', 'flag_deterministic']
VALID_KEYS += ['edu', 'measure', 'level', 'fixed_ambiguity', 'flag_ambiguity']
VALID_KEYS += ['max_draws', 'flag_precond', 'periods', 'types']
VALID_KEYS += ['flag_store', 'flag_myopic', 'fixed_delta', 'precond_type']


def process_constraints(dict_, constr, paras_fixed, paras_bounds):
    """ Check and process constraints.
    """

    # Check request
    _check_constraints(constr)

    # Replace path to dataset used for estimation
    if 'file_est' in constr.keys():
        # Checks
        assert isinstance(constr['file_est'], str)
        # Replace in initialization files
        dict_['ESTIMATION']['file'] = constr['file_est']

    # Replace interpolation
    if 'flag_interpolation' in constr.keys():
        # Checks
        assert (constr['flag_interpolation'] in [True, False])
        # Replace in initialization files
        dict_['INTERPOLATION']['flag'] = constr['flag_interpolation']

    # Replace number of periods
    if 'points' in constr.keys():
        # Extract objects
        points = constr['points']
        # Checks
        assert (isinstance(points, int))
        assert (points > 0)
        # Replace in initialization files
        dict_['INTERPOLATION']['points'] = points

    # Replace number of iterations
    if 'maxfun' in constr.keys():
        # Extract objects
        maxfun = constr['maxfun']
        # Checks
        assert (isinstance(maxfun, int))
        assert (maxfun >= 0)
        # Replace in initialization files
        dict_['ESTIMATION']['maxfun'] = maxfun

    # Replace education
    if 'edu' in constr.keys():
        # Extract objects
        start, max_ = constr['edu']
        # Checks
        assert (isinstance(start, int))
        assert (start > 0)
        assert (isinstance(max_, int))
        assert (max_ > start)
        # Replace in initialization file
        dict_['EDUCATION']['start'] = start
        dict_['EDUCATION']['max'] = max_

    # Replace measure of ambiguity
    if 'measure' in constr.keys():
        # Extract object
        measure = constr['measure']
        # Checks
        assert measure in ['kl', 'abs']
        # Replace in initialization file
        dict_['AMBIGUITY']['measure'] = measure

    # Replace level of ambiguity
    if 'level' in constr.keys():
        # Extract object
        level = constr['level']
        # Checks
        assert isinstance(level, float)
        assert level >= 0.0
        # Replace in initialization file
        dict_['AMBIGUITY']['coeffs'] = [level]
        dict_['AMBIGUITY']['bounds'] = [get_valid_bounds('amb', level)]

    # Treat level of ambiguity as fixed in an estimation
    if 'flag_ambiguity' in constr.keys():
        # Checks
        assert (constr['flag_ambiguity'] in [True, False])
        # Replace in initialization files
        if constr['flag_ambiguity']:
            value = np.random.uniform(0.01, 1.0)
            dict_['AMBIGUITY']['coeffs'] = [value]
            dict_['AMBIGUITY']['bounds'] = [get_valid_bounds('amb', value)]
        else:
            dict_['AMBIGUITY']['coeffs'] = [0.00]
            dict_['AMBIGUITY']['bounds'] = [get_valid_bounds('amb', 0.00)]

    # Treat level of ambiguity as fixed in an estimation
    if 'fixed_ambiguity' in constr.keys():
        # Checks
        assert (constr['fixed_ambiguity'] in [True, False])
        # Replace in initialization files
        dict_['AMBIGUITY']['fixed'] = [constr['fixed_ambiguity']]

    # Treat the discount rate as fixed in an estimation.
    if 'fixed_delta' in constr.keys():
        # Checks
        assert (constr['fixed_delta'] in [True, False])
        # Replace in initialization files
        dict_['BASICS']['fixed'] = [constr['fixed_delta']]

    # Replace version
    if 'version' in constr.keys():
        # Extract objects
        version = constr['version']
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
    if 'max_draws' in constr.keys():
        # Extract objects
        max_draws = constr['max_draws']
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
    if 'flag_parallelism' in constr.keys():
        # Extract objects
        flag_parallelism = constr['flag_parallelism']
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

    # Replace store attribute
    if 'flag_store' in constr.keys():
        # Extract objects
        flag_store = constr['flag_store']
        # Checks
        assert (flag_store in [True, False])
        # Replace in initialization file
        dict_['SOLUTION']['store'] = str(flag_store)

    # Replace number of periods
    if 'periods' in constr.keys():
        # Extract objects
        periods = constr['periods']
        # Checks
        assert (isinstance(periods, int))
        assert (periods > 0)
        # Replace in initialization files
        dict_['BASICS']['periods'] = periods

    # Replace discount factor
    if 'flag_myopic' in constr.keys():
        # Extract object
        assert ('delta' not in constr.keys())
        assert (constr['flag_myopic'] in [True, False])
        # Replace in initialization files
        if constr['flag_myopic']:
            dict_['BASICS']['coeffs'] = [0.0]
            dict_['BASICS']['bounds'] = [get_valid_bounds('delta', 0.00)]
        else:
            value = np.random.uniform(0.01, 1.0)
            dict_['BASICS']['coeffs'] = [value]
            dict_['BASICS']['bounds'] = [get_valid_bounds('amb', value)]

    # No random component to rewards
    if 'flag_deterministic' in constr.keys():
        # Checks
        assert (constr['flag_deterministic'] in [True, False])
        # Replace in initialization files
        if constr['flag_deterministic']:
            dict_['SHOCKS']['coeffs'] = [0.0] * 10

    # Number of agents
    if 'agents' in constr.keys():
        # Extract object
        num_agents = constr['agents']
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
    if 'flag_estimation' in constr.keys():
        # Checks
        assert (constr['flag_estimation'] in [True, False])
        # Replace in initialization files
        if constr['flag_estimation']:
            dict_['flag_store'] = False
            dict_['ESTIMATION']['maxfun'] = int(np.random.choice(range(6),
                p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
            dict_['PRECONDITIONING']['type'] = \
                np.random.choice(['gradient', 'identity', 'magnitudes'],
                    p=[0.1, 0.4, 0.5])

            # Ensure that a valid estimator is selected in the case that a
            # free parameter has bounds.
            for i, para_fixed in enumerate(paras_fixed):
                if para_fixed:
                    continue
                if any(item is not None for item in paras_bounds[i]):
                    if dict_['PROGRAM']['version'] == 'FORTRAN':
                        dict_['ESTIMATION']['optimizer'] = 'FORT-BOBYQA'
                    else:
                        dict_['ESTIMATION']['optimizer'] = 'SCIPY-LBFGSB'
                    break

    # It is important that these two constraints are imposed after
    # flag_estimation. Otherwise, they might be overwritten.
    if 'flag_precond' in constr.keys():
        # Extract objects
        flag_precond = constr['flag_precond']
        # Checks
        assert (flag_precond in [True, False])
        # Replace in initialization file
        if flag_precond:
            dict_['PRECONDITIONING']['type'] = 'gradient'
        else:
            dict_['PRECONDITIONING']['type'] = 'identity'

    if 'precond_type' in constr.keys():
        # Extract objects
        precond_type = constr['precond_type']
        # Checks
        assert (precond_type in ['identity', 'gradient', 'magnitudes'])
        # Replace in initialization file
        dict_['PRECONDITIONING']['type'] = precond_type

    return dict_


def _check_constraints(constr):
    """ Check that there are no conflicting constraints imposed.
    """
    # Check all specifie dconstraints
    for key_ in constr.keys():
        assert key_ in VALID_KEYS

    # Address incompatibility issues
    keys = constr.keys()

    if 'flag_myopic' in keys:
        assert 'delta' not in keys

    if 'flag_estimation' in keys:
        assert 'maxfun' not in keys
        assert 'flag_precond' not in keys

    if 'flag_ambiguity' in keys:
        assert 'level' not in keys

    if 'agents' in keys:
        assert 'max_draws' not in keys

    if 'flag_precond' in keys:
        assert 'precond_type' not in keys

    cond = ('flag_parallelism' in keys) and ('version' in keys)
    cond = cond and constr['flag_parallelism']
    if cond:
        assert constr['version'] == 'FORTRAN'
