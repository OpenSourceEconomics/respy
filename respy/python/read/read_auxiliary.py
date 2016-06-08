# standard library
import numpy as np

import sys
import os

# project library
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
STRUCTURE['SIMULATION'] = ['agents', 'seed', 'file']

STRUCTURE['ESTIMATION'] = ['draws', 'optimizer', 'maxiter', 'seed']
STRUCTURE['ESTIMATION'] += ['tau', 'file', 'agents']

STRUCTURE['PROGRAM'] = ['debug', 'version', 'parallelism', 'procs']
STRUCTURE['INTERPOLATION'] = ['apply', 'points']
STRUCTURE['SCIPY-BFGS'] = ['gtol', 'epsilon']
STRUCTURE['SCIPY-POWELL'] = ['maxfun', 'xtol', 'ftol']
STRUCTURE['FORT-NEWUOA'] = ['maxfun', 'npt', 'rhobeg', 'rhoend']
STRUCTURE['FORT-BFGS'] = ['maxiter', 'stpmx', 'gtol']


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
            if flag in ['debug', 'parallelism']:
                assert (value in [True, False])
            if flag in ['version']:
                assert (value in ['FORTRAN', 'PYTHON'])
                if value == 'FORTRAN':
                    fname = EXEC_DIR + '/resfort_scalar'
                    assert os.path.exists(fname)
            if flag in ['procs']:
                assert isinstance(value, int)
                assert value > 0

            if flag in ['parallelism'] and value:
                fname = EXEC_DIR + '/resfort_parallel_master'
                assert os.path.exists(fname)

        if group == 'SIMULATION':
            if flag in ['agents', 'seed']:
                assert isinstance(value, int)
                assert (value > 0)
            if flag in ['file']:
                assert isinstance(value, str)

        if group == 'INTERPOLATION':
            if flag in ['apply']:
                assert (value in [True, False])
            if flag in ['points']:
                assert isinstance(value, int)
            if flag in ['points']:
                assert (value > 0)

    except AssertionError:

        msg = '\n Misspecified initialization file (group, flag): '
        msg += group + ', ' + flag + '\n'
        sys.exit(msg)
