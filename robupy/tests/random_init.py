""" Module that contains the functions for the random generation of an
    initialization file.
"""

# standard library
import numpy as np

''' Module-specific Parameters
'''
MAX_AGENTS = 1000
MAX_DRAWS = 100
MAX_PERIODS = 4

''' Public Function
'''


def generate_init(constraints=None):
    """ Get a random initialization file.
    """

    # Antibugging. This interface is using a sentinal value.
    if constraints is not None:
        assert (isinstance(constraints, dict))

    dict_ = generate_random_dict(constraints)

    print_random_dict(dict_)

    # Finishing.
    return dict_


''' Private Functions
'''


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

    # Ambiguity (with temporary constraints)
    dict_['AMBIGUITY'] = dict()
    dict_['AMBIGUITY']['measure'] = np.random.choice(['absolute', 'kl'])
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

    # PROGRAM
    dict_['PROGRAM'] = {}
    dict_['PROGRAM']['debug'] = np.random.choice(['True', 'False'])
    dict_['PROGRAM']['version'] = np.random.choice(['FORTRAN', 'F2PY',
                                                    'PYTHON'])

    # SIMULATION
    dict_['SIMULATION'] = {}
    dict_['SIMULATION']['seed'] = np.random.random_integers(1, 10000)
    dict_['SIMULATION']['agents'] = np.random.random_integers(1, MAX_AGENTS)

    # Shocks
    cov = np.identity(4)
    for i, val in enumerate(np.random.uniform(0.05, 1, 4)):
        cov[i, i] = val
    dict_['SHOCKS'] = cov

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

    # Replace debugging level
    if 'debug' in constraints.keys():
        # Extract objects
        debug = constraints['debug']
        # Checks
        assert (debug in ['True', 'False'])
        # Replace in initialization file
        dict_['SOLUTION']['debug'] = debug

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
    if 'delta' in constraints.keys():
        # Extract object
        delta = constraints['delta']
        # Checks
        assert (np.isfinite(delta))
        assert (delta >= 0.0)
        # Replace in initialization file
        dict_['BASICS']['delta'] = delta

    # No random component to payoffs
    if 'eps_zero' in constraints.keys():
        # Checks
        assert (constraints['eps_zero'] is True)
        # Replace in initialization files
        dict_['SHOCKS'] = np.zeros((4, 4))

    # Ambiguity only of a particular type
    if 'measure' in constraints.keys():
        # Extract object
        measure = constraints['measure']
        # Checks
        assert (measure in ['kl', 'absolute'])
        # Replace in initialization files
        dict_['AMBIGUITY']['measure'] = measure

    # Finishing.
    return dict_


def print_random_dict(dict_):
    """ Print initialization dictionary to file. The different formatting
    makes the file rather involved.
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

            if flag in ['HOME']:

                file_.write(' HOME \n\n')

                str_ = ' {0:<15} {1:15.2f} \n'

                file_.write(str_.format('int', dict_[flag]['int']))

                file_.write('\n')

            if flag in ['SOLUTION', 'AMBIGUITY', 'SIMULATION', 'PROGRAM']:

                str_ = ' {0:<15} {1:<15} \n'

                file_.write(' ' + flag.upper() + '\n\n')

                for keys_ in dict_[flag]:
                    file_.write(str_.format(keys_, dict_[flag][keys_]))

                file_.write('\n')

            if flag in ['SHOCKS']:

                str_ = ' {0:15.2f} {1:15.2f} {2:15.2f} {3:15.2f}\n'

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

        str_ = ' {0:<15} {1:15.2f} {2:15.2f} \n'

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
