import statsmodels.api as sm
from scipy.stats import norm

import numpy as np
import pytest
import scipy
import sys
import os

from respy.python.solve.solve_auxiliary import get_predictions

from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import read_draws
from codes.auxiliary import write_draws
from respy.python.estimate.estimate_auxiliary import get_optim_paras

from respy.python.solve.solve_auxiliary import logging_solution
from respy.python.solve.solve_auxiliary import get_future_value
from respy.python.solve.solve_auxiliary import get_endogenous_variable
from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_constants import IS_FORTRAN

from respy.fortran.interface import resfort_interface
from respy.python.solve.solve_python import pyth_solve
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.evaluate.evaluate_python import pyth_evaluate
from respy.python.estimate.estimate_python import pyth_criterion

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR

from respy.python.solve.solve_auxiliary import pyth_create_state_space

from respy.python.solve.solve_auxiliary import pyth_calculate_payoffs_systematic

from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import get_simulated_indicator
from respy.python.solve.solve_auxiliary import get_exogenous_variables

from respy import solve
from respy import RespyCls

# Edit of PYTHONPATH required for PYTHON 2 as no __init__.py in tests
# subdirectory. If __init__.py is added, the path resolution for PYTEST
# breaks down.
if IS_FORTRAN:
    sys.path.insert(0, TEST_RESOURCES_DIR)
    import f2py_interface as fort_debug


@pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ Compare the evaluation of the criterion function for the ambiguity
        optimization and the simulated expected future value between the FORTRAN
        and PYTHON implementations. These tests are set up a separate test case
        due to the large setup cost to construct the ingredients for the interface.
        """
        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        respy_obj = solve(respy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, mapping_state_idx, \
        periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
        edu_max, delta, model_paras, is_debug = \
            dist_class_attributes(respy_obj,
                'periods_payoffs_systematic', 'states_number_period',
                'mapping_state_idx', 'periods_emax', 'num_periods',
                'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                'delta', 'model_paras', 'is_debug')

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Evaluation of simulated expected future values
        args = (num_periods, num_draws_emax, period, k, draws_standard,
            payoffs_systematic, edu_max, edu_start, periods_emax, states_all,
            mapping_state_idx, delta)

        py = get_future_value(*args)
        f90 = fort_debug.wrapper_get_future_value(*args)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    def test_2(self):
        """ Compare results between FORTRAN and PYTHON of selected
        hand-crafted functions. In test_97() we test FORTRAN implementations
        against PYTHON intrinsic routines.
        """

        for _ in range(25):

            # Create grid of admissible state space values.
            num_periods = np.random.randint(1, 15)
            edu_start = np.random.randint(1, 5)
            edu_max = edu_start + np.random.randint(1, 5)

            # Prepare interface
            min_idx = min(num_periods, (edu_max - edu_start + 1))

            # FORTRAN
            args = (num_periods, edu_start, edu_max, min_idx)
            fort_a, fort_b, fort_c, fort_d = \
                fort_debug.f2py_create_state_space(*args)
            py_a, py_b, py_c, py_d = pyth_create_state_space(*args)

            # Ensure equivalence
            for obj in [[fort_a, py_a], [fort_b, py_b], [fort_c, py_c], [fort_d, py_d]]:
                np.testing.assert_allclose(obj[0], obj[1])

        for _ in range(100):

            # Draw random request for testing purposes
            num_covars = np.random.randint(2, 10)
            num_agents = np.random.randint(100, 1000)
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

            # Check coefficient of determination and the standard errors.
            py = [results.rsquared, results.bse]
            f90 = fort_debug.wrapper_get_pred_info(endog, f90, exog,
                num_agents, num_covars)
            for i in range(2):
                np.testing.assert_almost_equal(py[i], f90[i])

    def test_3(self):
        """ Compare results between FORTRAN and PYTHON of selected functions.
        """
        for _ in range(10):

            # Draw random requests for testing purposes.
            num_draws_emax = np.random.randint(2, 1000)
            dim = np.random.randint(1, 6)

            matrix = (np.random.multivariate_normal(np.zeros(dim),
                np.identity(dim), dim))
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

            # Random normal deviates. This only tests the interface, requires
            # visual inspection in IPYTHON notebook as well.
            fort_debug.wrapper_standard_normal(num_draws_emax)

            # Clipping values below and above bounds.
            num_values = np.random.randint(1, 10000)
            lower_bound = np.random.randn()
            upper_bound = lower_bound + np.random.ranf()
            values = np.random.normal(size=num_values)

            f90 = fort_debug.wrapper_clip_value(values, lower_bound,
                upper_bound, num_values)
            py = np.clip(values, lower_bound, upper_bound)

            np.testing.assert_almost_equal(py, f90)

    def test_4(self):
        """ Testing the core functions of the solution step for the equality
        of results between the PYTHON and FORTRAN implementations.
        """

        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        write_interpolation_grid('test.respy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, model_paras, num_draws_emax, \
            seed_emax, is_debug, delta, is_interpolated, num_points_interp, = \
                dist_class_attributes(respy_obj,
                    'num_periods', 'edu_start', 'edu_max', 'min_idx',
                    'model_paras', 'num_draws_emax', 'seed_emax', 'is_debug',
                    'delta', 'is_interpolated', 'num_points_interp')

        # Auxiliary objects
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
            dist_model_paras(model_paras, is_debug)

        # Check the state space creation.
        args = (num_periods, edu_start, edu_max, min_idx)
        pyth = pyth_create_state_space(*args)
        f2py = fort_debug.f2py_create_state_space(*args)
        for i in range(4):
            np.testing.assert_allclose(pyth[i], f2py[i])

        # Carry some results from the state space creation for future use.
        states_all, states_number_period = pyth[:2]
        mapping_state_idx, max_states_period = pyth[2:]

        # Cutting to size
        states_all = states_all[:, :max(states_number_period), :]

        # Check calculation of systematic components of payoffs.
        args = (num_periods, states_number_period, states_all, edu_start,
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)
        pyth = pyth_calculate_payoffs_systematic(*args)
        f2py = fort_debug.f2py_calculate_payoffs_systematic(*args)
        np.testing.assert_allclose(pyth, f2py)

        # Carry some results from the systematic payoff calculation for
        # future use and create the required set of disturbances.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        periods_payoffs_systematic = pyth

        # Check backward induction procedure.
        args = (num_periods, max_states_period, periods_draws_emax,
            num_draws_emax, states_number_period, periods_payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, delta,
            is_debug, is_interpolated, num_points_interp, shocks_cholesky)

        logging_solution('start')
        pyth = pyth_backward_induction(*args)
        logging_solution('stop')

        f2py = fort_debug.f2py_backward_induction(*args)
        np.testing.assert_allclose(pyth, f2py)

    def test_5(self):
        """ This methods ensures that the core functions yield the same
        results across implementations.
        """

        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        max_states_period = write_interpolation_grid('test.respy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, model_paras, num_draws_emax, seed_emax, is_debug, delta, is_interpolated, num_points_interp, is_myopic, num_agents_sim, num_draws_prob, tau, paras_fixed, num_procs = dist_class_attributes(
            respy_obj, 'num_periods', 'edu_start', 'edu_max', 'min_idx',
            'model_paras', 'num_draws_emax', 'seed_emax', 'is_debug', 'delta',
            'is_interpolated', 'num_points_interp', 'is_myopic', 'num_agents_sim',
            'num_draws_prob', 'tau', 'paras_fixed', 'num_procs')

        # Write out random components and interpolation grid to align the
        # three implementations.
        max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
        write_draws(num_periods, max_draws)
        periods_draws_emax = read_draws(num_periods, num_draws_emax)
        periods_draws_prob = read_draws(num_periods, num_draws_prob)
        periods_draws_sims = read_draws(num_periods, num_agents_sim)

        # Extract coefficients
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = dist_model_paras(
            model_paras, True)

        # Check the full solution procedure
        base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta)

        fort = resfort_interface(respy_obj, 'solve')
        pyth = pyth_solve(*base_args + (periods_draws_emax,))
        f2py = fort_debug.f2py_solve(*base_args + (periods_draws_emax, max_states_period))

        for alt in [f2py, fort]:
            for i in range(5):
                np.testing.assert_allclose(pyth[i], alt[i])

        # Distribute solution arguments for further use in simulation test.
        periods_payoffs_systematic, _, mapping_state_idx, periods_emax, states_all = pyth

        args = (periods_payoffs_systematic, mapping_state_idx, \
            periods_emax, states_all, shocks_cholesky, num_periods, edu_start,
            edu_max, delta, num_agents_sim, periods_draws_sims)

        pyth = pyth_simulate(*args)

        f2py = fort_debug.f2py_simulate(*args)
        np.testing.assert_allclose(pyth, f2py)

        data_array = pyth

        base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky,
         is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
         edu_start, is_debug, edu_max, min_idx, delta, data_array, num_agents_sim,
         num_draws_prob, tau)

        args = base_args + (periods_draws_emax, periods_draws_prob)
        pyth = pyth_evaluate(*args)

        args = base_args + (periods_draws_emax, periods_draws_prob)
        f2py = fort_debug.f2py_evaluate(*args)

        np.testing.assert_allclose(pyth, f2py)

        # Evaluation of criterion function
        x0 = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, 'all', paras_fixed, is_debug)

        args = (
        is_interpolated, num_draws_emax, num_periods, num_points_interp, is_myopic,
        edu_start, is_debug, edu_max, min_idx, delta, data_array, num_agents_sim,
        num_draws_prob, tau, periods_draws_emax, periods_draws_prob)

        pyth = pyth_criterion(x0, *args)
        f2py = fort_debug.f2py_criterion(x0, *args)
        np.testing.assert_allclose(pyth, f2py)

    def test_6(self):
        """ Further tests for the interpolation routines.
        """
        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        respy_obj = solve(respy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, mapping_state_idx, seed_prob, periods_emax, model_paras, num_periods, states_all, num_points_interp, edu_start, num_draws_emax, is_debug, edu_max, delta = dist_class_attributes(
            respy_obj, 'periods_payoffs_systematic', 'states_number_period',
            'mapping_state_idx', 'seed_prob', 'periods_emax', 'model_paras',
            'num_periods', 'states_all', 'num_points_interp', 'edu_start',
            'num_draws_emax', 'is_debug', 'edu_max', 'delta')

        # Auxiliary objects
        shocks_cholesky = dist_model_paras(model_paras, is_debug)[-1]

        # Add some additional objects required for the interfaces to the
        # functions.
        period = np.random.choice(range(num_periods))

        periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_prob,
            is_debug)

        draws_emax = periods_draws_emax[period, :, :]

        num_states = states_number_period[period]

        shifts = np.random.randn(4)

        # Slight modification of request which assures that the
        # interpolation code is working.
        num_points_interp = min(num_points_interp, num_states)

        # Get the IS_SIMULATED indicator for the subset of points which are
        # used for the predication model.
        args = (num_points_interp, num_states, period, is_debug)
        is_simulated = get_simulated_indicator(*args)

        # Construct the exogenous variables for all points of the state
        # space.
        args = (
        period, num_periods, num_states, delta, periods_payoffs_systematic, shifts,
        edu_max, edu_start, mapping_state_idx, periods_emax, states_all)

        py = get_exogenous_variables(*args)
        f90 = fort_debug.wrapper_get_exogenous_variables(*args)

        np.testing.assert_equal(py, f90)

        # Distribute validated results for further functions.
        exogenous, maxe = py

        # Construct endogenous variable so that the prediction model can be
        # fitted.
        args = (period, num_periods, num_states, delta,
            periods_payoffs_systematic, edu_max, edu_start,
            mapping_state_idx, periods_emax, states_all, is_simulated,
            num_draws_emax, maxe, draws_emax)

        py = get_endogenous_variable(*args)
        f90 = fort_debug.wrapper_get_endogenous_variable(*args)

        np.testing.assert_equal(py, replace_missing_values(f90))

        # Distribute validated results for further functions.
        endogenous = py

        # Get predictions for expected future values. We need to start the
        # record to set up the handler for record the output from the
        # prediction model.
        logging_solution('start')

        args = (endogenous, exogenous, maxe, is_simulated, num_points_interp,
            num_states, is_debug)

        py = get_predictions(*args)
        f90 = fort_debug.wrapper_get_predictions(*args[:-1])

        np.testing.assert_array_almost_equal(py, f90)

        logging_solution('stop')

    def test_7(self):
        """ This is a special test for auxiliary functions related to the
        interpolation setup.
        """
        # Impose constraints
        constr = dict()
        constr['periods'] = np.random.randint(2, 5)

        # Construct a random initialization file
        generate_init(constr)

        # Extract required information
        respy_obj = RespyCls('test.respy.ini')

        # Extract class attributes
        is_debug, num_periods = dist_class_attributes(respy_obj,
                'is_debug', 'num_periods')

        # Write out a grid for the interpolation
        max_states_period = write_interpolation_grid('test.respy.ini')

        # Draw random request for testing
        num_states = np.random.randint(1, max_states_period)
        candidates = list(range(num_states))

        period = np.random.randint(1, num_periods)
        num_points_interp = np.random.randint(1, num_states + 1)

        # Check function for random choice and make sure that there are no
        # duplicates.
        f90 = fort_debug.wrapper_random_choice(candidates, num_states, num_points_interp)
        np.testing.assert_equal(len(set(f90)), len(f90))
        np.testing.assert_equal(len(f90), num_points_interp)

        # Check the standard cases of the function.
        args = (num_points_interp, num_states, period, is_debug, num_periods)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)

        np.testing.assert_equal(len(f90), num_states)
        np.testing.assert_equal(np.all(f90) in [0, 1], True)

        # Test the standardization across PYTHON, F2PY, and FORTRAN
        # implementations. This is possible as we write out an interpolation
        # grid to disk which is used for both functions.
        base_args = (num_points_interp, num_states, period, is_debug)
        args = base_args
        py = get_simulated_indicator(*args)
        args = base_args + (num_periods, )
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)
        np.testing.assert_array_equal(f90, 1*py)
        os.unlink('interpolation.txt')

        # Special case where number of interpolation points are same as the
        # number of candidates. In that case the returned indicator
        # should be all TRUE.
        args = (num_states, num_states, period, True, num_periods)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)
        np.testing.assert_equal(sum(f90), num_states)
