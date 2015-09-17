""" This module contains the testing infrastructure for the refactoring of the
SLSQP optimization algorithm.
"""

# standard library
from numpy.testing.utils import assert_array_almost_equal
from scipy.optimize import rosen_der, rosen

import numpy as np

import shutil
import glob
import sys
import os


DEBUG_OPTIONS = ' -O2 -fimplicit-none  -Wall  -Wline-truncation ' \
                ' -Wcharacter-truncation  -Wsurprising  -Waliasing' \
                ' -Wimplicit-interface  -Wunused-parameter  -fwhole-file ' \
                ' -fcheck=all  -std=f2008  -pedantic  -fbacktrace'


# testing library
sys.path.insert(0, '/home/peisenha/robustToolbox/package/development/tests/random')
from modules.auxiliary import compile_package

# project library
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import *
from robupy.python.py.ambiguity import _divergence
from robupy.python.py.ambiguity import _criterion

from slsqp import _minimize_slsqp
from scipy.optimize import approx_fprime

from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

def compile_tools():
     # Cleanup function?
    for files in glob.glob('*.so'):
        os.unlink(files)


    # Cleaup
    for dir_ in ['include', 'lib']:
        try:
            shutil.rmtree(dir_)
        except:
            pass

        os.mkdir(dir_)




    # Create the SLSQP library'robufort_development.f90 needs removed after
    # moving the functions to the core robufort library. Is there a role for
    # this file later at all?
    files = ['robufort_program_constants.f90', 'robufort_auxiliary.f90',
             'robufort_core.f90', 'robufort_development.f90',
             'robufort_slsqp.f90']
    for file_ in files:
        #os.system('gfortran -c  -fPIC ' + DEBUG_OPTIONS +  ' ' + file_)
        os.system('gfortran -c  -fPIC ' + file_)

    os.system('gfortran -c   -fPIC --fixed-form original_slsqp.f')
    os.system('ar crs libslsqp_debug.a *.o *.mod')

    module_files = glob.glob('*.mod')
    for file_ in module_files:
        shutil.move(file_, 'include/')

    shutil.move('libslsqp_debug.a', 'lib/')

    # Compile interface
    os.system(
          'f2py3 -c -m  f2py_slsqp_debug f2py_interface_slsqp.f90 -Iinclude -Llib '
            '-lslsqp_debug')


def test_implementations():


    compile_tools()

    # Import
    import f2py_slsqp_debug as fort

    # Ensure recomputability
    np.random.seed(345)

    for _ in range(1000):
        print(_)

        # Sample basic test case
        is_upgraded = np.random.choice([True, False])
        maxiter = np.random.random_integers(1, 100)
        num_dim = np.random.random_integers(2, 4)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=num_dim)

        # Test the upgraded FORTRAN version against the original code. This is
        # expected to NEVER fail.
        f_upgraded = fort.wrapper_slsqp_debug(x0, True, maxiter, ftol, num_dim)
        f_original = fort.wrapper_slsqp_debug(x0, False, maxiter, ftol, num_dim)
        np.testing.assert_array_equal(f_upgraded, f_original)

        # Test the FORTRAN codes against the PYTHON implementation. This is
        # expected to fail sometimes due to differences in precision between the
        # two implementations. In particular, as updating steps of the optimizer
        # are very sensitive to just small differences in the derivative
        # information. The same functions are available as a FORTRAN
        # implementations.
        def debug_constraint_derivative(x):
            return np.ones(len(x))
        def debug_constraint_function(x):
            return np.sum(x) - 10.0

        # Setting up PYTHON SLSQP interface for constraints
        constraint = dict()
        constraint['type'] = 'ineq'
        constraint['args'] = ()
        constraint['fun'] = debug_constraint_function
        constraint['jac'] = debug_constraint_derivative

        # Evaluate both implementations
        f = fort.wrapper_slsqp_debug(x0, is_upgraded, maxiter, ftol, num_dim)
        py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
                   ftol=ftol,  constraints=constraint)['x']
        np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)



compile_tools(); compile_package('fast')

import f2py_slsqp_debug as fort
import robupy.python.f2py.f2py_debug as f2py_debug

cov = np.identity(4)
level = 0.2

constraint = dict()

constraint['type'] = 'ineq'
constraint['fun'] = _divergence
#constraint['jac'] = None
constraint['args'] = (cov, level)

#constraint['fun'] = test_constraint
#constraint['jac'] = test_constraint_derivative
#constraint['args'] = ()
np.random.seed(123)

print('\n\n\n')

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
eps_standard = np.random.multivariate_normal(np.zeros(4), np.identity(4),
                                             (num_draws,))

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

for _ in range(0):
    num_dim = 2
    x0 = np.random.normal(size=num_dim)
    eps = np.random.uniform(0.000000, 0.5)

    print(eps)
    py = approx_fprime(x0, _criterion, eps, *args)
    f = f2py_debug.wrapper_approx_derivative(x0, num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max,
            edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
            eps_cholesky, delta, debug, eps)


    np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

#sys.exit('\n working on derivative of criterion function \n')

np.random.seed(42453)

while True:

    num_dim = 2

    is_upgraded = np.random.choice([True, False])
    maxiter = np.random.random_integers(1, 100)
    ftol = np.random.uniform(0.000000, 1e-5)
    x0 = np.random.normal(size=num_dim)

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

    # Sampling of random period and admissable state index
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
                   ftol=ftol,  constraints=constraint)['x']

    f = fort.wrapper_slsqp_robufort(x0, maxiter, ftol, cov,
            level, num_draws, eps_standard, period, k, payoffs_ex_ante, edu_max,
            edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
            eps_cholesky, delta, debug, num_dim)

    np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    print("")
    print(py, f)
