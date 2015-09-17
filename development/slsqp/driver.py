""" This module contains the testing infrastructure for the refactoring of the
SLSQP optimization algorithm.
"""

# standard library
from scipy.optimize.slsqp import _minimize_slsqp

import numpy as np

import shutil
import glob
import sys
import os

# project library
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import *

from robupy.tests.random_init import generate_init

from robupy.python.py.ambiguity import _divergence
from robupy.python.py.ambiguity import _criterion


def compile_tools():

    for files in glob.glob('*.so'):
        os.unlink(files)

    for dir_ in ['include', 'lib']:
        try:
            shutil.rmtree(dir_)
        except:
            pass

        os.mkdir(dir_)

    # Create the SLSQP library'robufort_development.f90 needs removed after
    # moving the functions to the core robufort library.
    files = ['robufort_program_constants.f90', 'robufort_auxiliary.f90',
             'robufort_slsqp.f90', 'robufort_core.f90']
    for file_ in files:
        os.system('gfortran -c  -fPIC ' + file_)
    os.system('gfortran -c  -fPIC --fixed-format original_slsqp.f')

    os.system('ar crs libslsqp_debug.a *.o *.mod')

    module_files = glob.glob('*.mod')
    for file_ in module_files:
        shutil.move(file_, 'include/')

    shutil.move('libslsqp_debug.a', 'lib/')

    # Compile interface
    os.system(
          'f2py3 -c -m  f2py_slsqp_debug f2py_interface_slsqp.f90 -Iinclude -Llib '
            '-lslsqp_debug')

compile_tools()

import f2py_slsqp_debug as fort

np.random.seed(42453)

while True:

    num_dim = 2

    is_upgraded = np.random.choice([True, False])
    maxiter = np.random.random_integers(1, 100)
    ftol = np.random.uniform(0.000000, 1e-5)
    x0 = np.random.normal(size=num_dim)

    eps = 1e-6

    cov = np.identity(4)*np.random.normal(size=1)**2
    level = np.random.normal(size=1)**2

    # Setting up PYTHON SLSQP interface for constraints
    constraint = dict()
    constraint['type'] = 'ineq'
    constraint['args'] = (cov, level)
    constraint['fun'] = _divergence

    # Generate constraint periods
    constraints = dict()
    constraints['version'] = 'PYTHON'

    # Generate random initialization file
    generate_init(constraints)

    # Perform toolbox actions
    robupy_obj = read('test.robupy.ini')

    robupy_obj = solve(robupy_obj)

    # Extract relevant information
    periods_payoffs_ex_ante = robupy_obj.get_attr('periods_payoffs_ex_ante')

    states_number_period = robupy_obj.get_attr('states_number_period')

    mapping_state_idx = robupy_obj.get_attr('mapping_state_idx')

    periods_emax = robupy_obj.get_attr('periods_emax')

    eps_cholesky = robupy_obj.get_attr('eps_cholesky')

    num_periods = robupy_obj.get_attr('num_periods')

    states_all = robupy_obj.get_attr('states_all')

    num_draws = robupy_obj.get_attr('num_draws')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    delta = robupy_obj.get_attr('delta')

    debug = False

    # Sample disturbances
    eps_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws,))

    # Sampling of random period and admissible state index
    period = np.random.choice(range(num_periods))
    k = np.random.choice(range(states_number_period[period]))

    # Select ex ante payoffs
    payoffs_ex_ante = periods_payoffs_ex_ante[period, k, :]

    # Evaluation point
    x = np.random.random(size=2)

    args = (num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max,
        edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
        eps_cholesky, delta, debug)

    py = _minimize_slsqp(_criterion, x0, args, maxiter=maxiter,
                   ftol=ftol, constraints=constraint)['x']

    f = fort.wrapper_slsqp_robufort(x0, maxiter, ftol, eps, num_draws,
            eps_standard, period, k, payoffs_ex_ante, edu_max, edu_start,
            mapping_state_idx, states_all, num_periods, periods_emax,
            eps_cholesky, delta, debug, cov, level, num_dim)

    np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    print("")
    print(py, f)
