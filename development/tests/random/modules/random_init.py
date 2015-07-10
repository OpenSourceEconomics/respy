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


def generate_init(constraints={}):
    """ Get a random initialization file.
    """
    dict_ = generate_random_dict(constraints)
        
    print_random_dict(dict_)
    
    # Finishing.
    return dict_

''' Private Functions
'''


def generate_random_dict(constraints={}):
    """ Draw random dictionary instance that can be processed into an
        initialization file.
    """
    # Initialize container
    dict_ = dict()

    # Basics
    dict_['BASICS'] = {}
    dict_['BASICS']['agents'] = np.random.random_integers(1, MAX_AGENTS)
    dict_['BASICS']['periods'] = np.random.random_integers(1, MAX_PERIODS)
    dict_['BASICS']['delta'] = np.random.random()

    # Ambiguity (with temporary constraints)
    dict_['AMBIGUITY'] = dict()
    dict_['AMBIGUITY']['measure'] =  np.random.choice(['absolute', 'kl'])
    dict_['AMBIGUITY']['level'] = np.random.choice([0.00, np.random.uniform()])
    dict_['AMBIGUITY']['para'] = 'both'

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
    dict_['EDUCATION']['max'] = np.random.random_integers(dict_['EDUCATION']['start'] + 1, 20)

    # Computation
    dict_['COMPUTATION'] = {}
    dict_['COMPUTATION']['draws'] = np.random.random_integers(1, MAX_DRAWS)
    dict_['COMPUTATION']['seed'] = np.random.random_integers(1, 10000)
    dict_['COMPUTATION']['debug'] = np.random.choice(['True', 'False'])

    # Shocks
    cov = np.random.normal(size=16).reshape((4,4))
    dict_['SHOCKS'] = np.dot(cov, cov.T)

    # Replace debugging level
    if 'debug' in constraints.keys():
        # Extract objects
        debug = constraints['debug']
        # Checks
        assert (debug in ['True', 'False'])
        # Replace in initialization file
        dict_['COMPUTATION']['debug'] = debug

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

    # Finishing.
    return dict_

def print_random_dict(dict_):
    """ Print initialization dictionary to file.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))

    # Create initialization.
    with open('test.robupy.ini', 'w') as file_:

        for flag in dict_.keys():

            if flag in ['COMPUTATION', 'BASICS', 'HOME', 'AMBIGUITY']:

                str_ = ' {0:<15} {1:<15} \n'

                file_.write(' ' + flag.upper() + '\n\n')

                for keys_ in dict_[flag]:

                    file_.write(str_.format(keys_, dict_[flag][keys_]))

                file_.write('\n')

            if flag in ['SHOCKS']:

                str_ = ' {0:<15} {1:<15} {2:<15} {3:<15}\n'

                file_.write(' ' + flag.upper() + '\n\n')

                for j in range(4):

                    file_.write(str_.format(*dict_[flag][j,:]))

                file_.write('\n')

            if flag in ['EDUCATION']:

                str_ = ' {0:<15} {1:<15} \n'

                file_.write(' ' + flag.upper() + '\n\n')

                file_.write(str_.format('coeff', dict_[flag]['coeff'][0]))

                file_.write(str_.format('coeff', dict_[flag]['coeff'][1]))

                file_.write('\n')

                file_.write(str_.format('int', dict_[flag]['int']))

                file_.write('\n')

                file_.write(str_.format('start', dict_[flag]['start']))

                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

    # Adding WORK
    with open('test.robupy.ini', 'a') as file_:

            str_ = ' {0:<15} {1:<15} {2:<15} \n'

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