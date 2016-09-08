import sys
import numpy as np
import pytest
from pandas.util.testing import assert_frame_equal
import pandas as pd

from codes.random_init import generate_random_dict

from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.estimate.estimate_auxiliary import get_optim_paras
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.estimate.estimate_python import pyth_criterion
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_auxiliary import read_draws
from respy.python.solve.solve_python import pyth_solve
from respy.fortran.interface import resfort_interface
from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_init
from codes.auxiliary import write_draws

from respy import RespyCls
from respy import simulate
from respy import estimate

# Edit of PYTHONPATH required for PYTHON 2 as no __init__.py in tests
# subdirectory. If __init__.py is added, the path resolution for PYTEST
# breaks down.
if IS_FORTRAN:
    sys.path.insert(0, TEST_RESOURCES_DIR)
    import f2py_interface as fort_debug


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This test ensures that using the ambiguity functionality with a
        tiny level yields the same results as using the risk functionality
        directly.

        """
        constr = dict()
        constr['maxfun'] = 0
        constr['flag_parallelism'] = False

        init_dict = generate_init(constr)

        base_val = None
        for level in [0.00, 0.00000000001]:

            init_dict['AMBIGUITY']['coeffs'] = [level]

            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')

            simulate(respy_obj)
            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val)

    def test_2(self):
        """ This test ensures that it does not matter which version runs
        the ambiguity codes.
        """

        max_draws = np.random.randint(10, 100)

        constr = dict()
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['level'] = np.random.uniform()
        constr['maxfun'] = 0

        init_dict = generate_init(constr)

        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)
        write_interpolation_grid('test.respy.ini')

        versions = ['PYTHON']
        if IS_FORTRAN:
            versions += ['FORTRAN']

        base_val = None
        for version in versions:
            init_dict['PROGRAM']['version'] = version

            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')

            simulate(respy_obj)
            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_3(self):
        """ This methods ensures that the core functions yield the same
        results across implementations in the case of ambiguity. The same
        function for the risk-only case is available in the test_f2py.py
        battery.
        """

        # Generate random initialization file
        constr = dict()
        constr['level'] = np.random.uniform(0.01, 0.99)
        generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        respy_obj = simulate(respy_obj)

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        max_states_period = write_interpolation_grid('test.respy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, model_paras, num_draws_emax, \
        is_debug, delta, is_interpolated, num_points_interp, is_myopic, \
        num_agents_sim, num_draws_prob, tau, paras_fixed, seed_sim, \
        measure, num_agents_est, states_number_period, \
        optimizer_options, derivatives = dist_class_attributes(respy_obj,
            'num_periods', 'edu_start',
            'edu_max', 'min_idx', 'model_paras', 'num_draws_emax',
            'is_debug', 'delta', 'is_interpolated', 'num_points_interp',
            'is_myopic', 'num_agents_sim', 'num_draws_prob', 'tau',
            'paras_fixed', 'seed_sim', 'measure',
            'num_agents_est', 'states_number_period',
            'optimizer_options', 'derivatives')

        fort_slsqp_maxiter = optimizer_options['FORT-SLSQP']['maxiter']
        fort_slsqp_ftol = optimizer_options['FORT-SLSQP']['ftol']
        fort_slsqp_eps = optimizer_options['FORT-SLSQP']['eps']

        # Write out random components and interpolation grid to align the
        # three implementations.
        max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
        write_draws(num_periods, max_draws)
        periods_draws_emax = read_draws(num_periods, num_draws_emax)
        periods_draws_prob = read_draws(num_periods, num_draws_prob)
        periods_draws_sims = read_draws(num_periods, num_agents_sim)

        # Extract coefficients
        level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky = \
            dist_model_paras(model_paras, True)

        # Check the full solution procedure
        base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, is_interpolated, num_points_interp,
            num_draws_emax, num_periods, is_myopic, edu_start, is_debug,
            edu_max, min_idx, delta, periods_draws_emax, measure, level)

        fort, _ = resfort_interface(respy_obj, 'simulate')
        py = pyth_solve(*base_args + (optimizer_options,))
        f2py = fort_debug.f2py_solve(*base_args + (max_states_period,
                                                   fort_slsqp_maxiter,
                                                   fort_slsqp_ftol, fort_slsqp_eps))

        for alt in [f2py, fort]:
            for i in range(5):
                np.testing.assert_allclose(py[i], alt[i])

        # Distribute solution arguments for further use in simulation test.
        periods_rewards_systematic, _, mapping_state_idx, periods_emax, \
        states_all = py

        args = (periods_rewards_systematic, mapping_state_idx, periods_emax,
                states_all, shocks_cholesky, num_periods, edu_start, edu_max,
                delta, num_agents_sim, periods_draws_sims, seed_sim)

        py = pyth_simulate(*args)

        f2py = fort_debug.f2py_simulate(*args)
        np.testing.assert_allclose(py, f2py)

        # Is is very important to cut the data array down to the size of the
        # estimation sample.
        data_array = py[:num_agents_est * num_periods, :]

        args = (periods_rewards_systematic, mapping_state_idx,
                periods_emax, states_all, shocks_cholesky, data_array,
                periods_draws_prob, delta, tau, edu_start, edu_max,
                num_periods, num_draws_prob)

        py = pyth_contributions(*args)
        f2py = fort_debug.f2py_contributions(*args + (num_agents_est,))
        np.testing.assert_allclose(py, f2py)

        # Evaluation of criterion function
        x0 = get_optim_paras(level, coeffs_a, coeffs_b, coeffs_edu, coeffs_home,
            shocks_cholesky, 'all', paras_fixed, is_debug)

        args = (is_interpolated, num_draws_emax, num_periods,
                num_points_interp, is_myopic, edu_start, is_debug, edu_max,
                delta, data_array, num_draws_prob, tau,
                periods_draws_emax, periods_draws_prob, states_all,
                states_number_period, mapping_state_idx, max_states_period,
                measure)

        py = pyth_criterion(x0, *args + (optimizer_options,))
        f2py = fort_debug.f2py_criterion(x0, *args + (
            fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps))
        np.testing.assert_allclose(py, f2py)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_4(self):
        """ Testing the equality of an evaluation of the criterion function for
        a random request. This test focuses on the risk-only case.
        """
        # Run evaluation for multiple random requests.
        is_deterministic = np.random.choice([True, False], p=[0.10, 0.9])
        is_interpolated = np.random.choice([True, False], p=[0.10, 0.9])
        is_myopic = np.random.choice([True, False], p=[0.10, 0.9])
        max_draws = np.random.randint(10, 100)

        # Generate random initialization file
        constr = dict()
        constr['is_deterministic'] = is_deterministic
        constr['flag_parallelism'] = False
        constr['is_myopic'] = is_myopic
        constr['max_draws'] = max_draws
        constr['maxfun'] = 0
        constr['level'] = np.random.uniform(0.01, 0.99)

        init_dict = generate_random_dict(constr)

        # The use of the interpolation routines is a another special case.
        # Constructing a request that actually involves the use of the
        # interpolation routine is a little involved as the number of
        # interpolation points needs to be lower than the actual number of
        # states. And to know the number of states each period, I need to
        # construct the whole state space.
        if is_interpolated:
            # Extract from future initialization file the information
            # required to construct the state space. The number of periods
            # needs to be at least three in order to provide enough state
            # points.
            num_periods = np.random.randint(3, 6)
            edu_start = init_dict['EDUCATION']['start']
            edu_max = init_dict['EDUCATION']['max']
            min_idx = min(num_periods, (edu_max - edu_start + 1))

            max_states_period = pyth_create_state_space(num_periods, edu_start,
                edu_max, min_idx)[3]

            # Updates to initialization dictionary that trigger a use of the
            # interpolation code.
            init_dict['BASICS']['periods'] = num_periods
            init_dict['INTERPOLATION']['flag'] = True
            init_dict['INTERPOLATION']['points'] = \
                np.random.randint(10, max_states_period)

        # Print out the relevant initialization file.
        print_init_dict(init_dict)

        # Write out random components and interpolation grid to align the
        # three implementations.
        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)
        write_interpolation_grid('test.respy.ini')

        # Clean evaluations based on interpolation grid,
        base_val, base_data = None, None

        for version in ['PYTHON', 'FORTRAN']:
            respy_obj = RespyCls('test.respy.ini')

            # Modify the version of the program for the different requests.
            respy_obj.unlock()
            respy_obj.set_attr('version', version)
            respy_obj.lock()

            # Solve the model
            respy_obj = simulate(respy_obj)

            # This parts checks the equality of simulated dataset for the
            # different versions of the code.
            data_frame = pd.read_csv('data.respy.dat', delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the
            # criterion function.
            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-05,
                atol=1e-06)

            # We know even more for the deterministic case.
            if constr['is_deterministic']:
                assert (crit_val in [-1.0, 0.0])

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_5(self):
        """ This test ensures that the evaluation of the criterion function
        at the starting value is identical between the different versions.
        This test focuses on the risk-only case.
        """

        max_draws = np.random.randint(10, 100)

        # Generate random initialization file
        constr = dict()
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['flag_interpolation'] = False
        constr['level'] = np.random.uniform(0.01, 0.99)
        constr['maxfun'] = 0

        # Generate random initialization file
        init_dict = generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Simulate a dataset
        simulate(respy_obj)

        # Iterate over alternative implementations
        base_x, base_val = None, None

        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)

        for version in ['FORTRAN', 'PYTHON']:

            respy_obj.unlock()

            respy_obj.set_attr('version', version)

            respy_obj.lock()

            x, val = estimate(respy_obj)

            # Check for the returned parameters.
            if base_x is None:
                base_x = x
            np.testing.assert_allclose(base_x, x)

            # Check for the value of the criterion function.
            if base_val is None:
                base_val = val
            np.testing.assert_allclose(base_val, val)
