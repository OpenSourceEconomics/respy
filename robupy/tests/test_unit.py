""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
from scipy.optimize.slsqp import _minimize_slsqp
from scipy.optimize import approx_fprime
from scipy.optimize import rosen_der
from scipy.optimize import rosen
from scipy.stats import norm

import statsmodels.api as sm
import numpy as np

import pytest
import scipy

# testing library
from codes.auxiliary import distribute_model_description
from codes.auxiliary import write_interpolation_grid

from robupy import simulate
from robupy import solve
from robupy import read

from robupy.python.py.python_library import get_payoffs

from robupy.python.solve_python import solve_python_bare

from robupy.tests.codes.random_init import generate_init

from robupy.python.py.ambiguity import get_payoffs_ambiguity
from robupy.python.py.auxiliary import simulate_emax
from robupy.python.py.ambiguity import _divergence
from robupy.python.py.ambiguity import _criterion

from robupy.auxiliary import opt_get_model_parameters
from robupy.auxiliary import opt_get_optim_parameters
from robupy.auxiliary import distribute_model_paras
from robupy.auxiliary import replace_missing_values
from robupy.auxiliary import create_draws

import robupy.python.py.python_library as py_lib

''' Main
'''

@pytest.mark.usefixtures('fresh_directory', 'set_seed', 'supply_resources')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This function compares the results from the payoff functions across
        implementations.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_debug as fort_debug

        for _ in range(10):

            # Impose constraints
            constraints = dict()
            constraints['measure'] = 'kl'

            # Generate random initialization file
            generate_init(constraints)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')
            robupy_obj = solve(robupy_obj)

            # Extract class attributes
            periods_payoffs_systematic, states_number_period, mapping_state_idx, \
            is_deterministic, is_ambiguous, periods_emax, model_paras, \
            num_periods, states_all, num_draws_emax, edu_start, is_debug,\
            edu_max, measure, delta, level = \
                distribute_model_description(robupy_obj,
                    'periods_payoffs_systematic', 'states_number_period',
                    'mapping_state_idx', 'is_deterministic', 'is_ambiguous',
                    'periods_emax', 'model_paras', 'num_periods', 'states_all',
                    'num_draws_emax', 'edu_start', 'is_debug', 'edu_max',
                    'measure', 'delta', 'level')

            # Extract auxiliary objects
            shocks_cholesky = model_paras['shocks_cholesky']
            shocks = model_paras['shocks']

            # Iterate over a couple of admissible points
            for _ in range(10):

                # Select random points
                period = np.random.choice(range(num_periods))
                k = np.random.choice(range(states_number_period[period]))

                # Finalize extraction of ingredients
                payoffs_systematic = periods_payoffs_systematic[period, k, :]
                draws_emax = np.random.sample((num_draws_emax, 4))

                # Extract payoffs using PYTHON and FORTRAN codes.
                py = get_payoffs(num_draws_emax, draws_emax, period, k,
                        payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                        states_all, num_periods, periods_emax, delta, is_debug,
                        shocks, level, is_ambiguous, measure, is_deterministic,
                        shocks_cholesky)

                f90 = fort_debug.wrapper_get_payoffs(num_draws_emax,
                        draws_emax, period, k, payoffs_systematic,
                        edu_max, edu_start, mapping_state_idx, states_all,
                        num_periods, periods_emax, delta, is_debug, shocks,
                        level, is_ambiguous, measure, is_deterministic,
                        shocks_cholesky)

                # Compare returned array on expected future values, ex post
                # payoffs, and future payoffs.
                for i in range(3):
                    np.testing.assert_array_almost_equal(py[i], f90[i])

    def test_2(self):
        """ This test compares the functions calculating the payoffs under
        ambiguity.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_debug as fort_debug

        # Iterate over random test cases
        for _ in range(10):

            # Generate constraint periods
            constraints = dict()
            constraints['level'] = 0.0
            constraints['version'] = 'PYTHON'
            constraints['measure'] = 'kl'

            # Generate random initialization file
            generate_init(constraints)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')
            robupy_obj = solve(robupy_obj)

            # Extract class attributes
            periods_payoffs_systematic, states_number_period, mapping_state_idx, \
            is_deterministic, periods_emax, num_periods, model_paras, \
            states_all, num_draws_emax, edu_start, edu_max, measure, \
            delta, is_debug = \
                distribute_model_description(robupy_obj,
                    'periods_payoffs_systematic', 'states_number_period',
                    'mapping_state_idx', 'is_deterministic', 'periods_emax',
                    'num_periods', 'model_paras', 'states_all', 'num_draws_emax',
                    'edu_start', 'edu_max', 'measure', 'delta', 'is_debug')

            # Auxiliary objects
            shocks = model_paras['shocks']
            shocks_cholesky = model_paras['shocks_cholesky']

            # Sample draws
            draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

            # Sampling of random period and admissible state index
            period = np.random.choice(range(num_periods))
            k = np.random.choice(range(states_number_period[period]))

            # Select systematic payoffs
            payoffs_systematic = periods_payoffs_systematic[period, k, :]

            # Set up optimization task
            level = np.random.uniform(0.01, 1.00)

            args = [num_draws_emax, draws_standard, period, k, payoffs_systematic,
                edu_max, edu_start, mapping_state_idx, states_all, num_periods,
                periods_emax, is_debug, delta, shocks, level, measure,
                is_deterministic, shocks_cholesky]

            f = fort_debug.wrapper_get_payoffs_ambiguity(*args)[0]
            py = get_payoffs_ambiguity(*args)[0]

            np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    def test_3(self):
        """ This test case compares the results from the SLSQP implementations in
        PYTHON and FORTRAN for the actual optimization problem.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_debug as fort_debug

        maxiter = np.random.random_integers(1, 100)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=2)

        tiny = 1e-6

        shocks = np.identity(4)*np.random.normal(size=1)**2
        level = np.random.normal(size=1)**2

        # Setting up PYTHON SLSQP interface for constraints
        constraint = dict()
        constraint['type'] = 'eq'
        constraint['args'] = (shocks, level)
        constraint['fun'] = _divergence

        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')
        robupy_obj = solve(robupy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, mapping_state_idx, \
            periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
            edu_max, delta, is_debug, model_paras = \
                distribute_model_description(robupy_obj,
                    'periods_payoffs_systematic', 'states_number_period',
                    'mapping_state_idx', 'periods_emax', 'num_periods',
                    'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                    'delta', 'is_debug', 'model_paras')

        # Auxiliary objects
        shocks_cholesky = model_paras['shocks_cholesky']

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                                np.identity(4), (num_draws_emax,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        args = (num_draws_emax, draws_standard, period, k, payoffs_systematic, edu_max,
            edu_start, mapping_state_idx, states_all, num_periods, periods_emax,
            delta, shocks_cholesky)

        opt = _minimize_slsqp(_criterion, x0, args, maxiter=maxiter,
                       ftol=ftol, constraints=constraint)

        # Stabilization. This is done as part of the fortran implementation.
        if opt['success']:
            py = opt['x']
        else:
            py = x0

        f = fort_debug.wrapper_slsqp_robufort(x0, maxiter, ftol, tiny,
                num_draws_emax, draws_standard, period, k, payoffs_systematic,
                edu_max, edu_start, mapping_state_idx, states_all,
                num_periods, periods_emax, delta, is_debug, shocks, level,
                shocks_cholesky)

        # Check equality. If not equal up to the tolerance, also check
        # whether the result from the FORTRAN implementation is even better.
        try:
            np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)
        except AssertionError:
            if _criterion(f, *args) < _criterion(py, *args):
                pass
            else:
                raise AssertionError

    def test_4(self):
        """ This test case compare the results of a debugging setup for the SLSQP
        algorithm's PYTHON and FORTRAN implementation
        """
        # FORTRAN resources
        import robupy.tests.lib.f2py_testing as fort_test

        # Sample basic test case
        maxiter = np.random.random_integers(1, 100)
        num_dim = np.random.random_integers(2, 4)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=num_dim)

        # Evaluation of Rosenbrock function. We are using the FORTRAN version
        # in the development of the optimization routines.
        f90 = fort_test.wrapper_debug_criterion_function(x0, num_dim)
        py = rosen(x0)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        py = rosen_der(x0)
        f90 = fort_test.wrapper_debug_criterion_derivative(x0, len(x0))
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
        constraint['type'] = 'eq'
        constraint['args'] = ()
        constraint['fun'] = debug_constraint_function
        constraint['jac'] = debug_constraint_derivative

        # Evaluate both implementations
        f = fort_test.wrapper_slsqp_debug(x0, maxiter, ftol, num_dim)
        py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
                ftol=ftol,  constraints=constraint)['x']
        np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    def test_5(self):
        """ Compare the evaluation of the criterion function for the ambiguity
        optimization and the simulated expected future value between the FORTRAN
        and PYTHON implementations. These tests are set up a separate test case
        due to the large setup cost to construct the ingredients for the interface.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_debug as fort_debug

        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, mapping_state_idx, \
        periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
        edu_max, delta, model_paras = \
            distribute_model_description(robupy_obj,
                'periods_payoffs_systematic', 'states_number_period',
                'mapping_state_idx', 'periods_emax', 'num_periods',
                'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                'delta', 'model_paras')

        # Auxiliary objects
        shocks_cholesky = model_paras['shocks_cholesky']

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

        shocks_mean = np.random.normal(size=2)

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Evaluation point
        x = np.random.random(size=2)

        # Evaluation of simulated expected future values
        py, _, _ = simulate_emax(num_periods, num_draws_emax, period, k,
            draws_standard, payoffs_systematic, edu_max, edu_start,
            periods_emax, states_all, mapping_state_idx, delta,
            shocks_cholesky, shocks_mean)

        f90, _, _ = fort_debug.wrapper_simulate_emax(num_periods,
            num_draws_emax, period, k, draws_standard, payoffs_systematic,
            edu_max, edu_start, periods_emax, states_all, mapping_state_idx,
            delta, shocks_cholesky, shocks_mean)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Criterion function for the determination of the worst case outcomes
        args = (num_draws_emax, draws_standard, period, k, payoffs_systematic,
                edu_max, edu_start, mapping_state_idx, states_all, num_periods,
                periods_emax, delta, shocks_cholesky)

        py = _criterion(x, *args)
        f90 = fort_debug.wrapper_criterion(x, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Evaluation of derivative of criterion function
        tiny = np.random.uniform(0.000000, 0.5)

        py = approx_fprime(x, _criterion, tiny, *args)
        f90 = fort_debug.wrapper_criterion_approx_gradient(x, tiny, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    def test_6(self):
        """ Compare results between FORTRAN and PYTHON of selected
        hand-crafted functions. In test_97() we test FORTRAN implementations
        against PYTHON intrinsic routines.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_library as fort_lib
        import robupy.python.f2py.f2py_debug as fort_debug

        for _ in range(10):

            # Draw random request for testing purposes
            matrix = (np.random.multivariate_normal(np.zeros(4), np.identity(
                    4), 4))
            cov = np.dot(matrix, matrix.T)
            x = np.random.rand(2)
            level = np.random.random(1)
            tiny = np.random.rand()**2

            # Kullback-Leibler (KL) divergence
            py = _divergence(x, cov, level)
            f90 = fort_debug.wrapper_divergence(x, cov, level)
            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Gradient approximation of KL divergence
            py = approx_fprime(x, _divergence, tiny, cov, level)
            f90 = fort_debug.wrapper_divergence_approx_gradient(x, cov, level, tiny)
            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        for _ in range(25):

            # Create grid of admissible state space values.
            num_periods = np.random.random_integers(1, 15)
            edu_start = np.random.random_integers(1, 5)
            edu_max = edu_start + np.random.random_integers(1, 5)

            # Prepare interface
            min_idx = min(num_periods, (edu_max - edu_start + 1))

            # FORTRAN
            fort_a, fort_b, fort_c, fort_d = fort_lib.wrapper_create_state_space(
                    num_periods, edu_start, edu_max, min_idx)

            # PYTHON
            py_a, py_b, py_c, py_d = py_lib.create_state_space(num_periods,
                    edu_start, edu_max, min_idx)

            # Ensure equivalence
            for obj in [[fort_a, py_a], [fort_b, py_b], [fort_c, py_c], [fort_d, py_d]]:
                    np.testing.assert_allclose(obj[0], obj[1])

        for _ in range(100):

            # Draw random request for testing purposes
            num_covars = np.random.random_integers(2, 10)
            num_agents = np.random.random_integers(100, 1000)
            tiny = np.random.normal(size=num_agents)
            beta = np.random.normal(size=num_covars)

            # Generate sample
            exog = np.random.sample((num_agents, num_covars))
            exog[:, 0] = 1
            endog = np.dot(exog, beta) + tiny

            # Run statsmodels
            results = sm.OLS(endog, exog).fit()

            # Check parameters
            py = results.params
            f90 = fort_debug.wrapper_get_coefficients(endog, exog, num_covars,
                        num_agents)
            np.testing.assert_almost_equal(py, f90)

            # Check prediction
            py = results.predict(exog)
            f90 = fort_debug.wrapper_point_predictions(exog, f90, num_agents)
            np.testing.assert_almost_equal(py, f90)

            # Check coefficient of determination
            py = results.rsquared
            f90 = fort_debug.wrapper_get_r_squared(endog, f90, num_agents)
            np.testing.assert_almost_equal(py, f90)

    def test_7(self):
        """ Compare results between FORTRAN and PYTHON of selected functions. The
        file python/f2py/debug_interface.f90 provides the F2PY bindings.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_debug as fort_debug



        for _ in range(10):

            # Draw random requests for testing purposes.
            num_draws_emax = np.random.random_integers(2, 1000)
            dim = np.random.random_integers(1, 6)
            mean = np.random.uniform(-0.5, 0.5, (dim))

            matrix = (np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim))
            cov = np.dot(matrix, matrix.T)

            # PDF of normal distribution
            args = np.random.normal(size=3)
            args[-1] **= 2

            f90 = fort_debug.wrapper_normal_pdf(*args)
            py = norm.pdf(*args)

            np.testing.assert_almost_equal(py, f90)

            # Singular Value Decomposition
            py = scipy.linalg.svd(matrix)
            f90 = fort_debug.wrapper_svd(matrix, dim)

            for i in range(3):
                np.testing.assert_allclose(py[i], f90[i], rtol=1e-05, atol=1e-06)

            # Pseudo-Inverse
            py = np.linalg.pinv(matrix)
            f90 = fort_debug.wrapper_pinv(matrix, dim)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Inverse
            py = np.linalg.inv(cov)
            f90 = fort_debug.wrapper_inverse(cov, dim)
            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Determinant
            py = np.linalg.det(cov)
            f90 = fort_debug.wrapper_determinant(cov)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Trace
            py = np.trace(cov)
            f90 = fort_debug.wrapper_trace(cov)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Cholesky decomposition
            f90 = fort_debug.wrapper_cholesky(cov, dim)
            py = np.linalg.cholesky(cov)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Random normal deviates. This only tests the interface, requires
            # visual inspection in IPYTHON notebook as well.
            fort_debug.wrapper_standard_normal(num_draws_emax)
            fort_debug.wrapper_multivariate_normal(mean, cov, num_draws_emax, dim)

            # Clipping values below and above bounds.
            num_values = np.random.random_integers(1, 10000)
            lower_bound = np.random.randn()
            upper_bound = lower_bound + np.random.ranf()
            values = np.random.normal(size=num_values)

            f90 = fort_debug.wrapper_get_clipped_vector(values, lower_bound, upper_bound,
                                                 num_values)
            py = np.clip(values, lower_bound, upper_bound)

            np.testing.assert_almost_equal(py, f90)

    def test_8(self):
        """ Testing the equality of the core functions for random requests.
        """
        # FORTRAN resources
        import robupy.python.f2py.f2py_debug as fort_debug

        # Generate random initialization file
        generate_init()

        # Solve random request
        robupy_obj = read('test.robupy.ini')

        # Extract class attributes
        is_interpolated, is_deterministic, seed_emax, is_ambiguous, \
        model_paras, num_periods, num_points, edu_start, is_python, \
        is_myopic, num_draws_emax, is_debug, measure, edu_max, \
        min_idx, delta, level = \
                distribute_model_description(robupy_obj,
                    'is_interpolated', 'is_deterministic', 'seed_emax',
                    'is_ambiguous', 'model_paras', 'num_periods',
                    'num_points', 'edu_start', 'is_python', 'is_myopic',
                    'num_draws_emax', 'is_debug', 'measure', 'edu_max',
                    'min_idx', 'delta', 'level')

        # Distribute model parameters
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks, shocks_cholesky = \
                distribute_model_paras(model_paras, is_debug)

        # Get set of draws
        draws_emax = create_draws(num_periods, num_draws_emax,
                                                seed_emax, is_debug, 'emax',
                                                shocks_cholesky)

        # Align interpolation grid
        max_states_period = write_interpolation_grid('test.robupy.ini')

        # Baseline input arguments.
        base_args = [coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks,
            edu_max, delta, edu_start, is_debug, is_interpolated, level,
            measure, min_idx, num_draws_emax, num_periods, num_points,
            is_ambiguous, draws_emax, is_deterministic, is_myopic,
            shocks_cholesky]

        # Check for the equality of the solution routines.
        base = None
        for version in ['PYTHON', 'F2PY', 'FORTRAN']:
            if version in ['F2PY', 'PYTHON']:
                # Modifications to input arguments
                args = base_args + [is_python]
                # Get PYTHON/F2PY results
                ret_args = solve_python_bare(*args)
            else:
                # Modifications to input arguments
                args = base_args + [max_states_period]
                # Get FORTRAN results
                ret_args = fort_debug.wrapper_solve_fortran_bare(*args)
            # Collect baseline information
            if base is None:
                base = ret_args
            # Check results
            for j in range(7):
                np.testing.assert_equal(base[j], replace_missing_values(
                    ret_args[j]))

    def test_9(self):
        """ Testing ten admissible realizations of state space for the first
        three periods.
        """
        # Generate constraint periods
        constraints = dict()
        constraints['periods'] = np.random.randint(3, 5)

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        simulate(robupy_obj)

        # Distribute class attributes
        states_number_period = robupy_obj.get_attr('states_number_period')

        states_all = robupy_obj.get_attr('states_all')

        # The next hard-coded results assume that at least two more
        # years of education are admissible.
        edu_max = robupy_obj.get_attr('edu_max')
        edu_start = robupy_obj.get_attr('edu_start')

        if edu_max - edu_start < 2:
            return

        # The number of admissible states in the first three periods
        for j, number_period in enumerate([1, 4, 13]):
            assert (states_number_period[j] == number_period)

        # The actual realizations of admissible states in period one
        assert ((states_all[0, 0, :] == [0, 0, 0, 1]).all())

        # The actual realizations of admissible states in period two
        states = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]]
        states += [[1, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[1, j, :] == state).all())

        # The actual realizations of admissible states in period three
        states = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
        states += [[0, 0, 2, 1], [0, 1, 0, 0], [0, 1, 1, 0]]
        states += [[0, 1, 1, 1], [0, 2, 0, 0], [1, 0, 0, 0]]
        states += [[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]]
        states += [[2, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[2, j, :] == state).all())

    def test_10(self):
        """ Testing whether back-and-forth transformation have no effect.
        """
        for i in range(10):
            # Create random parameter vector
            base = np.random.uniform(size=26)
            x = base.copy()

            # Apply numerous transformations
            for j in range(10):
                args = opt_get_model_parameters(x, is_debug=True)
                x = opt_get_optim_parameters(*args, is_debug=True)

            # Checks
            np.testing.assert_allclose(base, x)
