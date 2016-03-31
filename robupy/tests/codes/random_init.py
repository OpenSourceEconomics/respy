""" This module contains the functions for the generation of random requests.
"""

# standard library
import numpy as np

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

    print_random_dict(dict_)

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

    # Basics
    dict_['BASICS'] = {}
    dict_['BASICS']['periods'] = np.random.random_integers(1, MAX_PERIODS)
    dict_['BASICS']['delta'] = np.random.random()

    # Operationalization of ambiguity
    dict_['AMBIGUITY'] = dict()
    dict_['AMBIGUITY']['measure'] = np.random.choice(['kl'])
    dict_['AMBIGUITY']['level'] = np.random.choice([0.00, np.random.uniform()])

    # Home
    dict_['HOME'] = dict()
    dict_['HOME']['int'] = np.random.uniform(-0.05, 0.05, 1)[0]

    # Occupation A
    dict_['A'] = dict()
    dict_['A']['coeff'] = np.random.uniform(-0.05, 0.05, 5)
    dict_['A']['int'] = np.random.uniform(-0.05, 0.05, 1)[0]

    # Occupation B
    dict_['B'] = dict()
    dict_['B']['coeff'] = np.random.uniform(-0.05, 0.05, 5)
    dict_['B']['int'] = np.random.uniform(-0.05, 0.05, 1)[0]

    # Education
    dict_['EDUCATION'] = dict()
    dict_['EDUCATION']['coeff'] = np.random.uniform(-0.05, 0.05, 2)
    dict_['EDUCATION']['int'] = np.random.uniform(-0.05, 0.05, 1)[0]

    dict_['EDUCATION']['start'] = np.random.random_integers(1, 10)
    dict_['EDUCATION']['max'] = np.random.random_integers(
        dict_['EDUCATION']['start'] + 1, 20)

    # SOLUTION
    dict_['SOLUTION'] = {}
    dict_['SOLUTION']['draws'] = np.random.random_integers(1, MAX_DRAWS)
    dict_['SOLUTION']['seed'] = np.random.random_integers(1, 10000)
    dict_['SOLUTION']['store'] = np.random.choice(['True', 'False'])

    # ESTIMATION
    dict_['ESTIMATION'] = {}
    dict_['ESTIMATION']['draws'] = np.random.random_integers(1, MAX_DRAWS)
    dict_['ESTIMATION']['seed'] = np.random.random_integers(1, 10000)
    dict_['ESTIMATION']['file'] = 'data.robupy'
    dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])
    dict_['ESTIMATION']['maxiter'] = np.random.random_integers(1, 10000)

    # PROGRAM
    dict_['PROGRAM'] = {}
    dict_['PROGRAM']['debug'] = 'True'
    dict_['PROGRAM']['version'] = np.random.choice(['FORTRAN', 'F2PY',
                                                    'PYTHON'])

    # SIMULATION
    dict_['SIMULATION'] = {}
    dict_['SIMULATION']['seed'] = np.random.random_integers(1, 10000)
    dict_['SIMULATION']['agents'] = np.random.random_integers(1, MAX_AGENTS)
    dict_['SIMULATION']['file'] = 'data.robupy'

    # SHOCKS
    shocks_cov = np.identity(4)
    for i, val in enumerate(np.random.uniform(0.05, 1, 4)):
        shocks_cov[i, i] = val
    dict_['SHOCKS'] = shocks_cov

    # INTERPOLATION
    dict_['INTERPOLATION'] = {}
    dict_['INTERPOLATION']['apply'] = np.random.choice([True, False])
    dict_['INTERPOLATION']['points'] = np.random.random_integers(10, 100)

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
        assert (max_draws > 0)
        # Replace in initialization file
        dict_['SIMULATION']['agents'] = np.random.random_integers(1, max_draws)
        dict_['ESTIMATION']['draws'] = np.random.random_integers(1, max_draws)
        dict_['SOLUTION']['draws'] = np.random.random_integers(1, max_draws)

    # Replace store attribute
    if 'store' in constraints.keys():
        # Extract objects
        store = constraints['store']
        # Checks
        assert (store in [True, False])
        # Replace in initialization file
        dict_['SOLUTION']['store'] = store

    # Replace level of ambiguity
    if 'level' in constraints.keys():
        # Extract objects
        level = constraints['level']
        # Checks
        assert (np.isfinite(level))
        assert (level >= 0.0)
        assert (isinstance(level, float))
        # Replace in initialization file
        dict_['AMBIGUITY']['level'] = level

    # Replace measure of ambiguity
    if 'measure' in constraints.keys():
        # Extract objects
        measure = constraints['measure']
        # Checks
        assert (measure in ['kl', 'absolute'])
        # Replace in initialization file
        dict_['AMBIGUITY']['measure'] = measure

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
            dict_['SHOCKS'] = np.zeros((4, 4))

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

    # Number of simulations for S-ML
    if 'sims' in constraints.keys():
        # Extract object
        num_draws_prob = constraints['sims']
        # Checks
        assert (num_draws_prob > 0)
        assert (isinstance(num_draws_prob, int))
        assert (np.isfinite(num_draws_prob))
        # Replace in initialization files
        dict_['ESTIMATION']['draws'] = num_draws_prob

    # Finishing.
    return dict_


def print_random_dict(dict_):
    """ Print initialization dictionary to file. The different formatting
    makes the file rather involved. The resulting initialization files are
    read by PYTHON and FORTRAN routines. Thus, the formatting with respect to
    the number of decimal places is rather small.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))

    # Create initialization.
    with open('test.robupy.ini', 'w') as file_:

        for flag in dict_.keys():

            if flag in ['BASICS']:

                file_.write(' BASICS \n\n')

                str_ = ' {0:<15} {1:<15} \n'

                file_.write(str_.format('periods', dict_[flag]['periods']))

                str_ = ' {0:<15} {1:15.2f} \n'

                file_.write(str_.format('delta', dict_[flag]['delta']))

                file_.write('\n')

            if flag in ['HOME', 'AMBIGUITY']:

                file_.write(' ' + flag.upper() + '\n\n')

                for keys_ in dict_[flag]:

                    str_ = ' {0:<15} {1:15.2f} \n'

                    # Special treatment of ambiguity measure. which is a simple
                    #  string.
                    if keys_ in ['measure']:
                        str_ = ' {0:<15} {1:<15} \n'

                    file_.write(str_.format(keys_, dict_[flag][keys_]))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION']:

                str_ = ' {0:<15} {1:<15} \n'

                file_.write(' ' + flag.upper() + '\n\n')

                for keys_ in dict_[flag]:
                    file_.write(str_.format(keys_, str(dict_[flag][keys_])))

                file_.write('\n')

            if flag in ['SHOCKS']:

                # Type conversion
                dict_[flag] = np.array(dict_[flag])

                str_ = ' {0:15.4f} {1:15.4f} {2:15.4f} {3:15.4f}\n'

                file_.write(' ' + flag.upper() + '\n\n')

                for j in range(4):
                    file_.write(str_.format(*dict_[flag][j, :]))

                file_.write('\n')

            if flag in ['EDUCATION']:
                str_ = ' {0:<15} {1:15.2f} \n'

                file_.write(' ' + flag.upper() + '\n\n')

                file_.write(str_.format('coeff', dict_[flag]['coeff'][0]))

                file_.write(str_.format('coeff', dict_[flag]['coeff'][1]))

                file_.write('\n')

                file_.write(str_.format('int', dict_[flag]['int']))

                file_.write('\n')

                str_ = ' {0:<15} {1:<15} \n'

                file_.write(str_.format('start', dict_[flag]['start']))

                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

    # Adding WORK
    with open('test.robupy.ini', 'a') as file_:

        str_ = ' {0:<15} {1:15.4f} {2:15.4f} \n'

        file_.write(' WORK \n\n')

        # Coefficient
        for j in range(5):
            line = ['coeff', dict_['A']['coeff'][j]]

            line += [dict_['B']['coeff'][j]]

            file_.write(str_.format(*line))

        file_.write('\n')

        # Intercept
        line = ['int', dict_['A']['int'], dict_['B']['int']]

        file_.write(str_.format(*line))

        file_.write('\n')

    # Write out a valid specification for the admissible optimizers.
    lines = ['SCIPY-BFGS', 'gtol    1e-05', 'epsilon 1.4901161193847656e-08']
    lines += [' ']
    lines += ['SCIPY-POWELL', 'xtol 0.0001', 'ftol 0.0001']

    with open('optimization.robupy.opt', 'a') as file_:
        str_ = ' {0:>25} \n'
        for line in lines:
            file_.write(str_.format(line))