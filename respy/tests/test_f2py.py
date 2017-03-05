import sys
import os

import statsmodels.api as sm
from scipy.stats import norm
import numpy as np
import pytest
import scipy

from respy.python.solve.solve_auxiliary import pyth_calculate_rewards_systematic
from respy.python.shared.shared_auxiliary import correlation_to_covariance
from respy.python.shared.shared_auxiliary import covariance_to_correlation
from respy.python.shared.shared_utilities import spectral_condition_number
from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.solve.solve_ambiguity import get_relevant_dependence
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import get_simulated_indicator
from respy.python.solve.solve_auxiliary import get_exogenous_variables
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.solve.solve_auxiliary import get_endogenous_variable
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.estimate.estimate_python import pyth_criterion
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.solve.solve_auxiliary import get_predictions
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.solve.solve_risk import construct_emax_risk
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_auxiliary import read_draws
from respy.python.shared.shared_constants import IS_F2PY
from respy.python.solve.solve_python import pyth_solve
from respy.fortran.interface import resfort_interface
from codes.auxiliary import write_interpolation_grid
from codes.auxiliary import simulate_observed
from codes.random_init import generate_init
from codes.auxiliary import write_draws

from respy import RespyCls

# Edit of PYTHONPATH required for PYTHON 2 as no __init__.py in tests
# subdirectory. If __init__.py is added, the path resolution for PYTEST
# breaks down.
if IS_F2PY:
    sys.path.insert(0, TEST_RESOURCES_DIR)
    import f2py_interface as fort_debug


@pytest.mark.skipif(not IS_F2PY, reason='No F2PY available')
@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self, flag_ambiguity=False):
        """ Compare the evaluation of the criterion function for the ambiguity
        optimization and the simulated expected future value between the FORTRAN
        and PYTHON implementations. These tests are set up a separate test case
        due to the large setup cost to construct the ingredients for the interface.
        """
        # Generate constraint periods
        constr = dict()
        constr['version'] = 'PYTHON'
        constr['flag_ambiguity'] = flag_ambiguity

        # Generate random initialization file
        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        respy_obj = simulate_observed(respy_obj)

        # Extract class attributes
        periods_rewards_systematic, states_number_period, mapping_state_idx, \
            periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
            edu_max, optim_paras = dist_class_attributes(respy_obj,
                'periods_rewards_systematic', 'states_number_period',
                'mapping_state_idx', 'periods_emax', 'num_periods',
                'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                'optim_paras')

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Evaluation of simulated expected future values
        base_args = (num_periods, num_draws_emax, period, k, draws_standard,
            rewards_systematic, edu_max, edu_start, periods_emax, states_all,
            mapping_state_idx)

        args = ()
        args += base_args + (optim_paras,)
        py = construct_emax_risk(*args)

        args = ()
        args += base_args + (optim_paras['delta'],)
        f90 = fort_debug.wrapper_construct_emax_risk(*args)
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
            rslts = []
            rslts += [[fort_a, py_a], [fort_b, py_b]]
            rslts += [[fort_c, py_c], [fort_d, py_d]]
            for obj in rslts:
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

            matrix = np.random.uniform(size=dim ** 2).reshape(dim, dim)
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

            f90 = fort_debug.wrapper_clip_value(values, lower_bound, upper_bound,
                num_values)
            py = np.clip(values, lower_bound, upper_bound)

            np.testing.assert_almost_equal(py, f90)

            # Spectral condition number
            py = spectral_condition_number(cov)
            fort = fort_debug.wrapper_spectral_condition_number(cov)
            np.testing.assert_almost_equal(py, fort)

    def test_4(self, flag_ambiguity=False):
        """ Testing the core functions of the solution step for the equality
        of results between the PYTHON and FORTRAN implementations.
        """

        # Generate random initialization file
        constr = dict()
        constr['flag_ambiguity'] = flag_ambiguity

        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        write_interpolation_grid('test.respy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, optim_paras, num_draws_emax, \
            seed_emax, is_debug, is_interpolated, num_points_interp, \
            ambi_spec, optimizer_options, file_sim = \
                dist_class_attributes(respy_obj, 'num_periods', 'edu_start',
                    'edu_max', 'min_idx', 'optim_paras', 'num_draws_emax',
                    'seed_emax', 'is_debug', 'is_interpolated',
                    'num_points_interp', 'ambi_spec', 'optimizer_options',
                    'file_sim')

        # Distribute variables for FORTRAN interface
        fort_slsqp_maxiter = optimizer_options['FORT-SLSQP']['maxiter']
        fort_slsqp_ftol = optimizer_options['FORT-SLSQP']['ftol']
        fort_slsqp_eps = optimizer_options['FORT-SLSQP']['eps']

        shocks_cholesky = optim_paras['shocks_cholesky']
        coeffs_home = optim_paras['coeffs_home']
        coeffs_edu = optim_paras['coeffs_edu']
        coeffs_a = optim_paras['coeffs_a']
        coeffs_b = optim_paras['coeffs_b']
        level = optim_paras['level']
        delta = optim_paras['delta']

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

        # Check calculation of systematic components of rewards.
        args = ()
        args += (num_periods, states_number_period, states_all, edu_start)
        args += (max_states_period, )

        pyth = pyth_calculate_rewards_systematic(*args + (optim_paras, ))

        args = args + (coeffs_a, coeffs_b, coeffs_edu, coeffs_home)
        f2py = fort_debug.f2py_calculate_rewards_systematic(*args)
        np.testing.assert_allclose(pyth, f2py)

        # Carry some results from the systematic rewards calculation for
        # future use and create the required set of disturbances.
        periods_draws_emax = create_draws(num_periods, num_draws_emax,
            seed_emax, is_debug)

        periods_rewards_systematic = pyth

        # Check backward induction procedure.
        base_args = (num_periods, False, max_states_period, periods_draws_emax,
            num_draws_emax, states_number_period, periods_rewards_systematic,
            edu_max, edu_start, mapping_state_idx, states_all,
            is_debug, is_interpolated, num_points_interp)

        args = ()
        args += base_args + (ambi_spec, optim_paras, optimizer_options)
        args += (file_sim, False)
        pyth, _ = pyth_backward_induction(*args)

        args = ()
        args += base_args + (ambi_spec['measure'], ambi_spec['mean'])
        args += (shocks_cholesky, level, delta, fort_slsqp_maxiter)
        args += (fort_slsqp_ftol, fort_slsqp_eps, file_sim, False)
        f2py = fort_debug.f2py_backward_induction(*args)
        np.testing.assert_allclose(pyth, f2py)

    def test_5(self, flag_ambiguity=False):
        """ This methods ensures that the core functions yield the same
        results across implementations.
        """
        # Generate random initialization file
        constr = dict()
        constr['flag_ambiguity'] = flag_ambiguity
        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        respy_obj = simulate_observed(respy_obj)

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        max_states_period = write_interpolation_grid('test.respy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, optim_paras, num_draws_emax, \
            is_debug, is_interpolated, num_points_interp, is_myopic, \
            num_agents_sim, num_draws_prob, tau, seed_sim, ambi_spec, \
            num_agents_est, states_number_period, optimizer_options, \
            file_sim = dist_class_attributes(respy_obj,
                'num_periods', 'edu_start', 'edu_max', 'min_idx',
                'optim_paras', 'num_draws_emax', 'is_debug',
                'is_interpolated', 'num_points_interp', 'is_myopic',
                'num_agents_sim', 'num_draws_prob', 'tau', 'seed_sim',
                'ambi_spec', 'num_agents_est', 'states_number_period',
                'optimizer_options', 'file_sim')

        # Distribute variables for FORTRAN interface
        fort_slsqp_maxiter = optimizer_options['FORT-SLSQP']['maxiter']
        fort_slsqp_ftol = optimizer_options['FORT-SLSQP']['ftol']
        fort_slsqp_eps = optimizer_options['FORT-SLSQP']['eps']

        shocks_cholesky = optim_paras['shocks_cholesky']
        coeffs_home = optim_paras['coeffs_home']
        coeffs_edu = optim_paras['coeffs_edu']
        coeffs_a = optim_paras['coeffs_a']
        coeffs_b = optim_paras['coeffs_b']
        level = optim_paras['level']
        delta = optim_paras['delta']

        # Write out random components and interpolation grid to align the
        # three implementations.
        max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
        write_draws(num_periods, max_draws)
        periods_draws_emax = read_draws(num_periods, num_draws_emax)
        periods_draws_prob = read_draws(num_periods, num_draws_prob)
        periods_draws_sims = read_draws(num_periods, num_agents_sim)

        # Check the full solution procedure
        base_args = (is_interpolated, num_points_interp, num_draws_emax,
            num_periods, is_myopic, edu_start, is_debug, edu_max, min_idx,
            periods_draws_emax)

        fort, _ = resfort_interface(respy_obj, 'simulate')

        args = ()
        args += base_args + (ambi_spec, optim_paras, file_sim)
        args += (optimizer_options, )
        py = pyth_solve(*args)

        args = ()
        args += base_args + (ambi_spec['measure'], ambi_spec['mean'])
        args += (coeffs_a, coeffs_b, coeffs_edu, coeffs_home)
        args += (shocks_cholesky, level, delta, file_sim)
        args += (fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps)
        args += (max_states_period, )
        f2py = fort_debug.f2py_solve(*args)

        for alt in [fort, f2py]:
            for i in range(5):
                np.testing.assert_allclose(py[i], alt[i], rtol=1e-6)

        # Distribute solution arguments for further use in simulation test.
        periods_rewards_systematic, _, mapping_state_idx, periods_emax, \
            states_all = py

        base_args = (periods_rewards_systematic, mapping_state_idx,
            periods_emax, states_all, num_periods, edu_start, edu_max,
            num_agents_sim, periods_draws_sims, seed_sim, file_sim)

        args = ()
        args += base_args + (optim_paras, )
        py = pyth_simulate(*args)

        args = ()
        args += base_args + (shocks_cholesky, delta)
        f2py = fort_debug.f2py_simulate(*args)
        np.testing.assert_allclose(py, f2py)

        # Is is very important to cut the data array down to the size of the
        # estimation sample.
        data_array = py[:num_agents_est * num_periods, :]

        base_args = (periods_rewards_systematic, mapping_state_idx,
            periods_emax, states_all, data_array, periods_draws_prob,
            tau, edu_start, edu_max, num_periods, num_draws_prob)

        args = ()
        args += base_args + (optim_paras, )
        py = pyth_contributions(*args)

        args = ()
        args += base_args + (shocks_cholesky, optim_paras['delta'])
        args += (num_agents_est,)
        f2py = fort_debug.f2py_contributions(*args)

        np.testing.assert_allclose(py, f2py)

        # Evaluation of criterion function
        x0 = get_optim_paras(optim_paras, 'all', is_debug)

        base_args = (is_interpolated, num_draws_emax, num_periods,
            num_points_interp, is_myopic, edu_start, is_debug, edu_max,
            data_array, num_draws_prob, tau, periods_draws_emax,
            periods_draws_prob, states_all, states_number_period,
            mapping_state_idx, max_states_period)

        args = ()
        args += base_args + (ambi_spec, optimizer_options, )
        py, _ = pyth_criterion(x0, *args)

        args = ()
        args += base_args + (ambi_spec['measure'], ambi_spec['mean'])
        args += (fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps)
        f2py = fort_debug.f2py_criterion(x0, *args)

        np.testing.assert_allclose(py, f2py)

    def test_6(self, flag_ambiguity=False):
        """ Further tests for the interpolation routines.
        """
        # Generate random initialization file
        constr = dict()
        constr['flag_ambiguity'] = flag_ambiguity
        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        respy_obj = simulate_observed(respy_obj)

        # Extract class attributes
        periods_rewards_systematic, states_number_period, mapping_state_idx, \
            seed_prob, periods_emax, num_periods, states_all, \
            num_points_interp, edu_start, num_draws_emax, is_debug, edu_max, \
            ambi_spec, optim_paras, optimizer_options, file_sim = \
                dist_class_attributes(respy_obj, 'periods_rewards_systematic',
                'states_number_period', 'mapping_state_idx', 'seed_prob',
                'periods_emax', 'num_periods', 'states_all',
                'num_points_interp', 'edu_start', 'num_draws_emax',
                'is_debug', 'edu_max', 'ambi_spec', 'optim_paras',
                'optimizer_options', 'file_sim')

        # Initialize containers
        i, j = num_periods, max(states_number_period)
        opt_ambi_details = np.tile(MISSING_FLOAT, (i, j, 7))

        shocks_cov = np.matmul(optim_paras['shocks_cholesky'], optim_paras['shocks_cholesky'].T)

        # Distribute variables for FORTRAN interface
        fort_slsqp_maxiter = optimizer_options['FORT-SLSQP']['maxiter']
        fort_slsqp_ftol = optimizer_options['FORT-SLSQP']['ftol']
        fort_slsqp_eps = optimizer_options['FORT-SLSQP']['eps']

        level = optim_paras['level']
        # Add some additional objects required for the interfaces to the
        # functions.
        period = np.random.choice(range(num_periods))

        periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_prob,
            is_debug)

        draws_emax_standard = periods_draws_emax[period, :, :]
        draws_emax_transformed = transform_disturbances(draws_emax_standard,
            np.tile(0, 4), optim_paras['shocks_cholesky'])

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
        base_args = (period, num_periods, num_states,
            periods_rewards_systematic, shifts, edu_max, edu_start,
            mapping_state_idx, periods_emax, states_all)

        args = ()
        args += base_args + (optim_paras,)
        py = get_exogenous_variables(*args)

        args = ()
        args += base_args + (optim_paras['delta'],)
        f90 = fort_debug.wrapper_get_exogenous_variables(*args)

        np.testing.assert_equal(py, f90)

        # Distribute validated results for further functions.
        exogenous, maxe = py

        # Construct endogenous variable so that the prediction model can be
        # fitted.
        base_args = (period, num_periods, num_states,
            periods_rewards_systematic, edu_max, edu_start,
            mapping_state_idx, periods_emax, states_all, is_simulated,
            num_draws_emax, maxe, draws_emax_standard, draws_emax_transformed)

        args = ()
        args += base_args + (ambi_spec, optim_paras, optimizer_options)
        args += (opt_ambi_details, )
        py, _ = get_endogenous_variable(*args)

        args = ()
        args += base_args + (shocks_cov, ambi_spec['measure'])
        args += (ambi_spec['mean'], level, optim_paras['delta'])
        args += (fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps)
        f90 = fort_debug.wrapper_get_endogenous_variable(*args)
        np.testing.assert_almost_equal(py, replace_missing_values(f90))

        # Distribute validated results for further functions.
        base_args = (py, exogenous, maxe, is_simulated)

        args = ()
        args += base_args + (file_sim, False)
        py = get_predictions(*args)

        args = ()
        args += base_args + (num_points_interp, num_states, file_sim, False)
        f90 = fort_debug.wrapper_get_predictions(*args)

        np.testing.assert_array_almost_equal(py, f90)

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
        args = (candidates, num_states, num_points_interp)
        f90 = fort_debug.wrapper_random_choice(*args)
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

    def test_8(self):
        """ We test the construction of the Cholesky decomposition against
        each other.
        """
        # Draw a random vector of parameters
        x = np.random.uniform(size=28)

        # Construct the Cholesky decompositions
        py = extract_cholesky(x, info=0)
        fort = fort_debug.wrapper_extract_cholesky(x)

        # Compare the results based on the two methods
        np.testing.assert_equal(fort, py)

    def test_9(self):
        """ We test the some of the functions that were added when improving
        the ambiguity set.
        """

        for _ in range(100):

            is_deterministic = np.random.choice([True, False], p=[0.1, 0.9])
            dim = np.random.randint(1, 6)

            if is_deterministic:
                cov = np.zeros((dim, dim))
            else:
                matrix = np.random.uniform(size=dim ** 2).reshape(dim, dim)
                cov = np.dot(matrix, matrix.T)

            if not is_deterministic:
                py = np.linalg.cholesky(cov)
                f90 = fort_debug.wrapper_get_cholesky_decomposition(cov, dim)
                np.testing.assert_almost_equal(py, f90)

            py = covariance_to_correlation(cov)
            f90 = fort_debug.wrapper_covariance_to_correlation(cov, dim)
            np.testing.assert_almost_equal(py, f90)

            corr, sd = covariance_to_correlation(cov), np.sqrt(np.diag(cov))
            py = correlation_to_covariance(corr, sd)
            f90 = fort_debug.wrapper_correlation_to_covariance(corr, sd)
            np.testing.assert_almost_equal(py, f90)

            # Now we also check that the back-and-forth transformations work.
            corr = fort_debug.wrapper_covariance_to_correlation(cov, dim)
            cov_cand = fort_debug.wrapper_correlation_to_covariance(corr, sd, dim)
            np.testing.assert_almost_equal(cov, cov_cand)

            corr = covariance_to_correlation(cov)
            cov_cand = correlation_to_covariance(corr, sd)
            np.testing.assert_almost_equal(cov, cov_cand)

        # We also check whether the construction of the candidate
        # covariance matrix during the worst-case determintation works
        # well. These functions only work for the 4x4 covariance matrix.
        for _ in range(100):
            is_deterministic = np.random.choice([True, False], p=[0.1, 0.9])
            mean = np.random.choice([True, False])

            if is_deterministic:
                cov = np.zeros((4, 4))
            else:
                matrix = np.random.uniform(size=16).reshape(4, 4)
                cov = np.dot(matrix, matrix.T)

            x0 = np.random.uniform(low=-1.0, high=1.0, size=2)
            if not mean:
                x0 = np.append(x0, np.random.uniform(low=0.0, high=1.0, size=2))

            py = get_relevant_dependence(cov, x0)
            f90 = fort_debug.wrapper_get_relevant_dependence(cov, x0)

            for i in range(2):
                np.testing.assert_almost_equal(py[i], f90[i])