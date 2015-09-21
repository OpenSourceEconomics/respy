""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal

from scipy.optimize.slsqp import _minimize_slsqp
from scipy.optimize import approx_fprime
from scipy.optimize import rosen_der
from scipy.optimize import rosen

import pandas as pd
import numpy as np

import sys
import os

# testing library
from modules.auxiliary import transform_robupy_to_restud
from modules.auxiliary import write_disturbances
from modules.auxiliary import build_f2py_testing
from modules.auxiliary import compile_package

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import read, solve, simulate

from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict
from robupy.tests.random_init import generate_init

from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import _divergence
from robupy.python.py.ambiguity import _criterion


''' Main
'''
def test_93():
    """ This test case compares the results from the SLSQP implementations in
    PYTHON and FORTRAN for the actual optimization problem.
    """
    # Ensure interface is available
    build_f2py_testing()
    import modules.f2py_testing as fort

    # Sample problem parameters
    for _ in range(10):
        print(_)
        maxiter = np.random.random_integers(1, 100)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=2)

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
                eps_cholesky, delta, debug, cov, level)

        # Check equality. If not equal up to the tolerance, I also check
        # whether the result from the FORTRAN implementation is even better.
        try:
            np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)
        except AssertionError:
            if _criterion(f) < _criterion(py):
                pass
            else:
                raise AssertionError

def test_94():
    """ This test case compare the results of a debugging setup for the SLSQP
    algorithm's PYTHON and FORTRAN implementation
    """
    # Ensure interface is available
    build_f2py_testing()
    import modules.f2py_testing as fort

    # Sample basic test case
    maxiter = np.random.random_integers(1, 100)
    num_dim = np.random.random_integers(2, 4)
    ftol = np.random.uniform(0.000000, 1e-5)
    x0 = np.random.normal(size=num_dim)

    # Evaluation of Rosenbrock function. We are using the FORTRAN version
    # in the development of the optimization routines.
    f90 = fort.wrapper_debug_criterion_function(x0, num_dim)
    py = rosen(x0)
    np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    py = rosen_der(x0)
    f90 = fort.wrapper_debug_criterion_derivative(x0, len(x0))
    np.testing.assert_allclose(py, f90[:-1], rtol=1e-05, atol=1e-06)

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
    f = fort.wrapper_slsqp_debug(x0, maxiter, ftol, num_dim)
    py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
            ftol=ftol,  constraints=constraint)['x']
    np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)


def test_95():
    """ Compare the evaluation of the criterion function for the ambiguity
    optimization and the simulated expected future value between the FORTRAN
    and PYTHON implementations. These tests are set up a separate test case
    due to the large setup cost to construct the ingredients for the interface.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(10):

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

        # Evaluation of simulated expected future values
        py, _, _ = simulate_emax(num_periods, num_draws, period, k,
                        eps_standard, payoffs_ex_ante, edu_max, edu_start,
                        periods_emax, states_all, mapping_state_idx, delta)

        f90, _, _ = fort.wrapper_simulate_emax(num_periods, num_draws,
                        period, k, eps_standard, payoffs_ex_ante, edu_max,
                        edu_start, periods_emax, states_all,
                        mapping_state_idx, delta)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Criterion function for the determination of the worst case outcomes
        args = (num_draws, eps_standard, period, k, payoffs_ex_ante,
                edu_max, edu_start, mapping_state_idx, states_all, num_periods,
                periods_emax, eps_cholesky, delta, debug)

        py = _criterion(x, *args)
        f90, _, _ = fort.wrapper_criterion(x, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Evaluation of derivative of criterion function
        eps = np.random.uniform(0.000000, 0.5)

        py = approx_fprime(x, _criterion, eps, *args)
        f90 = fort.wrapper_criterion_approx_gradient(x, eps, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)


def test_96():
    """ Compare results between FORTRAN and PYTHON of selected
    hand-crafted functions. In test_97() we test FORTRAN implementations
    against PYTHON intrinsic routines.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(1000):

        # Draw random request for testing purposes
        matrix = (np.random.multivariate_normal(np.zeros(4), np.identity(
            4), 4))
        cov = np.dot(matrix, matrix.T)
        x = np.random.rand(2)
        level = np.random.random(1)
        eps = np.random.rand()**2

        # Kullback-Leibler (KL) divergence
        py = _divergence(x, cov, level)
        f90 = fort.wrapper_divergence(x, cov, level)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Gradient approximation of KL divergence
        py = approx_fprime(x, _divergence, eps, cov, level)
        f90 = fort.wrapper_divergence_approx_gradient(x, cov, level, eps)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)


def test_97():
    """ Compare results between FORTRAN and PYTHON of selected functions. The
    file python/f2py/debug_interface.f90 provides the F2PY bindings.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(1000):

        # Draw random requests for testing purposes.
        num_draws = np.random.random_integers(2, 1000)
        dim = np.random.random_integers(1, 6)
        mean = np.random.uniform(-0.5, 0.5, (dim))

        matrix = (np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim))
        cov = np.dot(matrix, matrix.T)

        # Inverse
        py = np.linalg.inv(cov)
        f90 = fort.wrapper_inverse(cov, dim)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Determinant
        py = np.linalg.det(cov)
        f90 = fort.wrapper_determinant(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Trace
        py = np.trace(cov)
        f90 = fort.wrapper_trace(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Cholesky decomposition
        f90 = fort.wrapper_cholesky(cov, dim)
        py = np.linalg.cholesky(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Random normal deviates. This only tests the interface, requires
        # visual inspection in IPYTHON notebook as well.
        fort.wrapper_standard_normal(num_draws)
        fort.wrapper_multivariate_normal(mean, cov, num_draws, dim)


def test_98():
    """  Compare results from the RESTUD program and the ROBUPY package.
    """

    # Ensure that fast solution methods are available
    compile_package('fast')

    # Prepare RESTUD program
    os.chdir('modules')
    os.system(' gfortran -fcheck=bounds -o dp3asim dp3asim.f95 >'
              ' /dev/null 2>&1 ')
    os.remove('pei_additions.mod')
    os.remove('imsl_replacements.mod')
    os.chdir('../')

    # Impose some constraints on the initialization file which ensures that
    # the problem can be solved by the RESTUD code.
    constraints = dict()
    constraints['edu'] = (10, 20)
    constraints['level'] = 0.00
    constraints['debug'] = 'True'

    version = np.random.choice(['FORTRAN', 'F2PY', 'PYTHON'])
    constraints['version'] = version

    # Generate random initialization file. The RESTUD code uses the same random
    # draws for the solution and simulation of the model. Thus, the number of
    # draws is required to be less or equal to the number of agents.
    init_dict = generate_random_dict(constraints)

    num_agents = init_dict['SIMULATION']['agents']
    num_draws = init_dict['SOLUTION']['draws']
    if num_draws < num_agents:
        init_dict['SOLUTION']['draws'] = num_agents

    print_random_dict(init_dict)

    # Write out disturbances to align the three implementations.
    write_disturbances(init_dict)

    # Perform toolbox actions
    robupy_obj = read('test.robupy.ini')

    # This flag aligns the random components between the RESTUD program and
    # ROBUPY package. The existence of the file leads to the RESTUD program
    # to write out the random components.
    init_dict = robupy_obj.get_attr('init_dict')

    transform_robupy_to_restud(init_dict)

    # Solve model using RESTUD code.
    os.system('./modules/dp3asim > /dev/null 2>&1')

    # Solve model using ROBUPY package.
    solve(robupy_obj)

    # Compare the simulated datasets generated by the programs.
    py = pd.DataFrame(np.array(np.genfromtxt('data.robupy.dat',
            missing_values='.'), ndmin=2)[:, -4:])

    fort = pd.DataFrame(np.array(np.genfromtxt('ftest.txt',
            missing_values='.'), ndmin=2)[:, -4:])

    assert_frame_equal(py, fort)

    # Cleanup
    os.unlink('disturbances.txt')


def test_99():
    """ Testing whether the results from a fast and slow execution of the
    code result in identical simulate datasets.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    # Constraint to risk model
    constraints = dict()
    constraints['level'] = 0.0
    constraints['debug'] = 'True'

    # Just making sure that it also works for this special case.
    if np.random.choice([True, False]):
        constraints['eps_zero'] = True

    # Generate random initialization file. Since this exercise is only for
    # debugging purposes, the codes uses the same disturbances for the
    # simulation and solution of the model. Thus, the number of agents cannot
    # be larger than the number of draws.
    init_dict = generate_random_dict(constraints)

    num_agents = init_dict['SIMULATION']['agents']
    num_draws = init_dict['SOLUTION']['draws']
    if num_draws < num_agents:
        init_dict['SOLUTION']['draws'] = num_agents

    print_random_dict(init_dict)

    # Write out disturbances to align the three implementations.
    write_disturbances(init_dict)

    # Initialize containers
    base = None

    for version in ['PYTHON', 'F2PY', 'FORTRAN']:

        # Prepare initialization file
        init_dict['PROGRAM']['version'] = version

        print_random_dict(init_dict)

        # Simulate the ROBUPY package
        os.system('robupy-solve --model test.robupy.ini')

        # Load simulated data frame
        data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

        data_frame.to_csv(version + '.dat')

        # Compare
        if base is None:
            base = data_frame.copy()

        assert_frame_equal(base, data_frame)