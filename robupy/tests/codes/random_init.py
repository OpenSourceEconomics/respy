""" This module contains the functions for the generation of random requests.
"""

# standard library
import numpy as np

import os

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

    # The dictionary also contains the information whether parameters are
    # fixed during an estimation. We need to ensure that at least one
    # parameter is always free. Note, that we only sample 17 realizations
    # even though there are 26 parameters. The last entry decides whether the
    # full covariance matrix is fixed or not.
    paras_fixed = np.random.choice([True, False], 17).tolist()
    if sum(paras_fixed) == 17:
        paras_fixed[np.random.randint(0, 17)] = True

    # Sampling number of agents for the simulation. This is then used as the
    # upper bound for the dataset used in the estimation.
    num_agents_sim = np.random.randint(3, MAX_AGENTS)

    # Basics
    dict_['BASICS'] = {}
    dict_['BASICS']['periods'] = np.random.randint(1, MAX_PERIODS)
    dict_['BASICS']['delta'] = np.random.random()

    # Operationalization of ambiguity
    dict_['AMBIGUITY'] = dict()
    dict_['AMBIGUITY']['level'] = np.random.choice([0.00, np.random.uniform()])

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
    dict_['ESTIMATION']['file'] = 'data.robupy'
    dict_['ESTIMATION']['optimizer'] = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])
    dict_['ESTIMATION']['maxiter'] = np.random.randint(1, 10000)
    dict_['ESTIMATION']['tau'] = np.random.uniform(100, 500)

    # PROGRAM
    dict_['PROGRAM'] = {}
    dict_['PROGRAM']['debug'] = 'True'
    dict_['PROGRAM']['version'] = np.random.choice(['FORTRAN', 'F2PY',
                                                    'PYTHON'])

    # SIMULATION
    dict_['SIMULATION'] = {}
    dict_['SIMULATION']['seed'] = np.random.randint(1, 10000)
    dict_['SIMULATION']['agents'] = num_agents_sim
    dict_['SIMULATION']['file'] = 'data.robupy'

    # SHOCKS
    dict_['SHOCKS'] = dict()
    shocks = np.zeros(10)
    for i in [0, 4, 7, 9]:
        shocks[i] = np.random.uniform(0.05, 1)
    dict_['SHOCKS']['coeffs'] = shocks
    dict_['SHOCKS']['fixed'] = np.tile(paras_fixed[16:17], 10)

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


def format_opt_parameters(val, identifier, paras_fixed):
    """ This function formats the values depending on whether they are fixed
    during the optimization or not.
    """
    # Initialize baseline line
    line = ['coeff', val, ' ']
    if paras_fixed[identifier]:
        line[-1] = '!'

    # Finishing
    return line


def print_random_dict(dict_, file_name='test.robupy.ini'):
    """ Print initialization dictionary to file. The different formatting
    makes the file rather involved. The resulting initialization files are
    read by PYTHON and FORTRAN routines. Thus, the formatting with respect to
    the number of decimal places is rather small.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))

    paras_fixed = dict_['OCCUPATION A']['fixed'][:]
    paras_fixed += dict_['OCCUPATION B']['fixed'][:]
    paras_fixed += dict_['EDUCATION']['fixed'][:]
    paras_fixed += dict_['HOME']['fixed'][:]
    paras_fixed += [dict_['SHOCKS']['fixed'][0]][:]

    str_optim = ' {0:<10} {1:20.4f} {2:>5} \n'

    # Construct labels. This ensures that the initialization files alway look
    # identical.
    labels = ['BASICS', 'AMBIGUITY', 'OCCUPATION A', 'OCCUPATION B']
    labels += ['EDUCATION', 'HOME', 'SHOCKS',  'SOLUTION']
    labels += ['SIMULATION', 'ESTIMATION', 'PROGRAM', 'INTERPOLATION']
    labels += ['SCIPY-BFGS', 'SCIPY-POWELL']

    # Create initialization.
    with open(file_name, 'w') as file_:

        for flag in labels:
            if flag in ['BASICS']:

                file_.write(' BASICS \n\n')

                str_ = ' {0:<10} {1:>20} \n'
                file_.write(str_.format('periods', dict_[flag]['periods']))

                str_ = ' {0:<10} {1:20.4f} \n'
                file_.write(str_.format('delta', dict_[flag]['delta']))

                file_.write('\n')

            if flag in ['AMBIGUITY']:

                file_.write(' ' + flag.upper() + '\n\n')

                # This function can also be used to print out initialization
                # files that only work for the RESPY package.
                if flag not in dict_.keys():
                    continue

                for keys_ in dict_[flag]:

                    str_ = ' {0:<10} {1:20.4f} \n'

                    file_.write(str_.format(keys_, dict_[flag][keys_]))

                file_.write('\n')

            if flag in ['HOME']:

                file_.write(' ' + flag.upper() + '\n\n')

                val = dict_['HOME']['coeffs'][0]
                line = format_opt_parameters(val, 15, paras_fixed)
                file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SOLUTION', 'SIMULATION', 'PROGRAM', 'INTERPOLATION',
                        'ESTIMATION']:

                file_.write(' ' + flag.upper() + '\n\n')

                for keys_ in dict_[flag]:

                    if keys_ in ['tau']:
                        str_ = ' {0:<10} {1:20.0f} \n'
                        file_.write(str_.format(keys_, dict_[flag][keys_]))
                    else:
                        str_ = ' {0:<10} {1:>20} \n'
                        file_.write(str_.format(keys_, str(dict_[flag][keys_])))

                file_.write('\n')

            if flag in ['SHOCKS']:

                # Type conversion
                file_.write(' ' + flag.upper() + '\n\n')

                for i in range(10):
                    val = dict_['SHOCKS']['coeffs'][i]
                    line = format_opt_parameters(val, 16, paras_fixed)
                    file_.write(str_optim.format(*line))
                file_.write('\n')

            if flag in ['EDUCATION']:

                file_.write(' ' + flag.upper() + '\n\n')

                val = dict_['EDUCATION']['coeffs'][0]
                line = format_opt_parameters(val, 12, paras_fixed)
                file_.write(str_optim.format(*line))

                val = dict_['EDUCATION']['coeffs'][1]
                line = format_opt_parameters(val, 13, paras_fixed)
                file_.write(str_optim.format(*line))

                val = dict_['EDUCATION']['coeffs'][2]
                line = format_opt_parameters(val, 14, paras_fixed)
                file_.write(str_optim.format(*line))

                file_.write('\n')
                str_ = ' {0:<10} {1:>15} \n'
                file_.write(str_.format('start', dict_[flag]['start']))
                file_.write(str_.format('max', dict_[flag]['max']))

                file_.write('\n')

            if flag in ['OCCUPATION A', 'OCCUPATION B']:
                identifier = None
                if flag == 'OCCUPATION A':
                    identifier = 0
                if flag == 'OCCUPATION B':
                    identifier = 6

                file_.write(flag + '\n\n')

                # Coefficient
                for j in range(6):
                    val = dict_[flag]['coeffs'][j]
                    line = format_opt_parameters(val, identifier, paras_fixed)
                    identifier += 1

                    file_.write(str_optim.format(*line))

                file_.write('\n')

            if flag in ['SCIPY-POWELL', 'SCIPY-BFGS']:

                # This function can also be used to print out initialization
                # files without any optimization options. This is enough for
                # simulation tasks.
                if flag not in dict_.keys():
                    continue

                file_.write(' ' + flag.upper() + '\n\n')

                for keys_ in dict_[flag]:

                    if keys_ in ['maxfun']:
                        str_ = ' {0:<10} {1:>20} \n'
                        file_.write(str_.format(keys_, dict_[flag][keys_]))
                    else:
                        str_ = ' {0:<10} {1:20.4f} \n'
                        file_.write(str_.format(keys_, dict_[flag][keys_]))

                file_.write('\n')