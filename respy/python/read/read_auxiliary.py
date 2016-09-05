import numpy as np
import sys
import os

from respy.python.shared.shared_constants import EXEC_DIR

# Hard coded structure of admissible groups and flags in the
# initialization file.
STRUCTURE = dict()
STRUCTURE['BASICS'] = ['periods', 'delta']
STRUCTURE['OCCUPATION A'] = ['coeff']
STRUCTURE['OCCUPATION B'] = ['coeff']
STRUCTURE['EDUCATION'] = ['coeff', 'max', 'start']
STRUCTURE['HOME'] = ['coeff']
STRUCTURE['SHOCKS'] = ['coeff']
STRUCTURE['SOLUTION'] = ['draws', 'store', 'seed']
STRUCTURE['AMBIGUITY'] = ['coeff', 'measure']

STRUCTURE['SIMULATION'] = ['agents', 'seed', 'file']

STRUCTURE['ESTIMATION'] = ['draws', 'optimizer', 'maxfun', 'seed']
STRUCTURE['ESTIMATION'] += ['tau', 'file', 'agents']

STRUCTURE['DERIVATIVES'] = ['version', 'eps']
STRUCTURE['SCALING'] = ['flag', 'minimum']

STRUCTURE['PROGRAM'] = ['debug', 'version']
STRUCTURE['PARALLELISM'] = ['flag', 'procs']

STRUCTURE['INTERPOLATION'] = ['flag', 'points']
STRUCTURE['SCIPY-BFGS'] = ['gtol', 'maxiter']
STRUCTURE['SCIPY-POWELL'] = ['maxfun', 'xtol', 'ftol', 'maxiter']
STRUCTURE['FORT-NEWUOA'] = ['maxfun', 'npt', 'rhobeg', 'rhoend']
STRUCTURE['FORT-BFGS'] = ['maxiter', 'stpmx', 'gtol']
STRUCTURE['SCIPY-SLSQP'] = ['maxiter', 'ftol']
STRUCTURE['FORT-SLSQP'] = ['maxiter', 'ftol']


def check_line(group, flag, value):
    """ Check each line of the initialization file
    """
    # Check for admissible group/flag combinations
    assert (group in STRUCTURE.keys())
    assert (flag in STRUCTURE[group])

    # Check the values for the flags.
    try:

        if (group, flag) == ('BASICS', 'periods'):
            assert isinstance(value, int)
            assert (value > 0)

        for arg in ['OCCUPATION A', 'OCCUPATION B']:
            if (group, flag) in (arg, 'coeff'):
                assert isinstance(value, float)
                assert np.isfinite(value)

        for arg in ['max', 'start']:
            if (group, flag) in ('EDUCATION', arg):
                assert isinstance(value, int)
                assert np.isfinite(value)
                assert (value >= 0)

        if (group, flag) == ('HOME', 'coeff'):
            assert isinstance(value, float)
            assert np.isfinite(value)

        if group == 'SIMULATION':
            if flag in ['agents', 'draws', 'seed']:
                assert isinstance(value, int)
                assert (value > 0)
            if flag in ['file']:
                assert isinstance(value, str)
            if flag in ['tau']:
                assert isinstance(value, float)
                assert np.isfinite(float)
                assert (value > 0)

        if group == 'SOLUTION':
            if flag in ['draws', 'seed']:
                assert isinstance(value, int)
                assert np.isfinite(value)
            if flag in ['store']:
                assert (value in [True, False])

        if group == 'PROGRAM':
            if flag in ['debug']:
                assert (value in [True, False])
            if flag in ['version']:
                assert (value in ['FORTRAN', 'PYTHON'])
                if value == 'FORTRAN':
                    fname = EXEC_DIR + '/resfort_scalar'
                    assert os.path.exists(fname)
            if flag in ['procs']:
                assert isinstance(value, int)
                assert value > 0

        if group == 'PARALLELISM':
            if flag in ['flag'] and value:
                fname = EXEC_DIR + '/resfort_parallel_master'
                assert os.path.exists(fname)
            if flag in ['procs']:
                assert isinstance(value, int)
                assert value > 0

        if group == 'AMBIGUITY':
            if flag in ['level']:
                assert isinstance(value, float)
                assert (value >= 0)
            if flag in ['measure']:
                assert value in ['kl', 'abs']

        if group == 'SIMULATION':
            if flag in ['agents', 'seed']:
                assert isinstance(value, int)
                assert (value > 0)
            if flag in ['file']:
                assert isinstance(value, str)

        if group == 'DERIVATIVE':
            if flag in ['version']:
                assert (value in ['forward-differences'])
            if flag in ['eps']:
                assert isinstance(value, float)
                assert (value > 0)

        if group == 'SCALING':
            if flag in ['flag']:
                assert (value in [True, False])
            if flag in ['minimum']:
                assert isinstance(value, float)
                assert (value > 0)

        if group == 'INTERPOLATION':
            if flag in ['flag']:
                assert (value in [True, False])
            if flag in ['points']:
                assert isinstance(value, int)
                assert (value > 0)

    except AssertionError:

        msg = '\n Misspecified initialization file (group, flag): '
        msg += group + ', ' + flag + '\n'
        sys.exit(msg)
