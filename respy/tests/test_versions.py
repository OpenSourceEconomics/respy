from pandas.util.testing import assert_frame_equal
import numpy as np
import pandas as pd
import pytest

from respy.python.solve.solve_auxiliary import pyth_create_state_space
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import MIN_AMBIGUITY
from respy.python.shared.shared_constants import IS_FORTRAN
from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_random_dict
from codes.auxiliary import simulate_observed
from codes.auxiliary import compare_est_log
from codes.random_init import generate_init
from codes.auxiliary import write_draws
from respy import estimate
from respy import RespyCls


@pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self, flag_ambiguity=False):
        """ Testing the equality of an evaluation of the criterion function for
        a random request.
        """
        # Run evaluation for multiple random requests.
        is_deterministic = np.random.choice([True, False], p=[0.10, 0.9])
        is_interpolated = np.random.choice([True, False], p=[0.10, 0.9])
        is_myopic = np.random.choice([True, False], p=[0.10, 0.9])
        seed_observed = np.random.randint(10, 100)
        max_draws = np.random.randint(10, 100)

        # Generate random initialization file
        constr = dict()
        constr['is_deterministic'] = is_deterministic
        constr['flag_ambiguity'] = flag_ambiguity
        constr['flag_parallelism'] = False
        constr['is_myopic'] = is_myopic
        constr['max_draws'] = max_draws
        constr['maxfun'] = 0

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
            respy_obj = simulate_observed(respy_obj, seed=seed_observed)

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

    def test_2(self, flag_ambiguity=False):
        """ This test ensures that the evaluation of the criterion function
        at the starting value is identical between the different versions.
        """

        max_draws = np.random.randint(10, 100)

        # Generate random initialization file
        constr = dict()
        constr['flag_ambiguity'] = flag_ambiguity
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['flag_interpolation'] = False
        constr['maxfun'] = 0

        # Generate random initialization file
        init_dict = generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Simulate a dataset
        simulate_observed(respy_obj)

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

    @pytest.mark.slow
    def test_3(self):
        """ This test just locks in the evaluation of the criterion function
        for the original Keane & Wolpin data.
        """
        # Sample one task
        resources = ['kw_data_one.ini', 'kw_data_two.ini', 'kw_data_three.ini']
        fname = np.random.choice(resources)

        # Select expected result
        rslt = None
        if 'one' in fname:
            rslt = 0.261487735867433
        elif 'two' in fname:
            rslt = 1.126138097174159
        elif 'three' in fname:
            rslt = 1.895699121131644

        # Evaluate criterion function at true values.
        respy_obj = RespyCls(TEST_RESOURCES_DIR + '/' + fname)

        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)
        respy_obj.lock()

        simulate_observed(respy_obj, share_missing_obs=0, share_missing_wages=0)
        _, val = estimate(respy_obj)
        np.testing.assert_allclose(val, rslt)

    def test_4(self):
        """ Test the solution of deterministic model with ambiguity and
        interpolation. This test has the same result as in the absence of
        random variation in rewards, it does not matter whether the
        environment is ambiguous or not.
        """
        # Solve specified economy
        for version in ['FORTRAN', 'PYTHON']:
            respy_obj = RespyCls(TEST_RESOURCES_DIR + '/test_fifth.respy.ini')

            respy_obj.unlock()

            respy_obj.set_attr('version', version)

            respy_obj.lock()

            respy_obj = simulate_observed(respy_obj)

            # Assess expected future value
            val = respy_obj.get_attr('periods_emax')[0, :1]
            np.testing.assert_allclose(val, 88750)

            # Assess evaluation
            _, val = estimate(respy_obj)
            np.testing.assert_allclose(val, -1.0)

    def test_5(self):
        """ Test the solution of deterministic model without ambiguity,
        but with interpolation. As a deterministic model is requested,
        all versions should yield the same result without any additional effort.
        """
        # Solve specified economy
        for version in ['FORTRAN', 'PYTHON']:
            respy_obj = RespyCls(TEST_RESOURCES_DIR + '/test_fifth.respy.ini')

            respy_obj.unlock()

            respy_obj.set_attr('version', version)

            respy_obj.lock()

            respy_obj = simulate_observed(respy_obj)

            # Assess expected future value
            val = respy_obj.get_attr('periods_emax')[0, :1]
            np.testing.assert_allclose(val, 88750)

            # Assess evaluation
            _, val = estimate(respy_obj)
            np.testing.assert_allclose(val, -1.0)

    def test_6(self, flag_ambiguity=False):
        """ This test ensures that the logging looks exactly the same for the
        different versions.
        """

        seed_observed = np.random.random_integers(0, 100)
        max_draws = np.random.randint(10, 300)

        # Generate random initialization file
        constr = dict()
        constr['flag_ambiguity'] = flag_ambiguity
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['flag_interpolation'] = False
        constr['maxfun'] = 0

        # Generate random initialization file
        init_dict = generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')
        level = respy_obj.get_attr('model_paras')['level']
        file_sim = respy_obj.get_attr('file_sim')
        is_ambiguity = (level > MIN_AMBIGUITY)

        # Iterate over alternative implementations
        base_sol_log, base_est_info, base_est_log = None, None, None
        base_sim_log, base_amb_log = None, None

        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)

        for version in ['FORTRAN', 'PYTHON']:

            respy_obj.unlock()

            respy_obj.set_attr('version', version)

            respy_obj.lock()

            simulate_observed(respy_obj, seed=seed_observed)

            # Check for identical logging
            fname = file_sim + '.respy.sol'
            if base_sol_log is None:
                base_sol_log = open(fname, 'r').read()
            assert open(fname, 'r').read() == base_sol_log

            # Check for identical logging
            fname = file_sim + '.respy.sim'
            if base_sim_log is None:
                base_sim_log = open(fname, 'r').read()
            assert open(fname, 'r').read() == base_sim_log

            # Check for identical logging
            fname = file_sim + '.respy.amb'
            if is_ambiguity:
                if base_amb_log is None:
                    base_amb_log = open(fname, 'r').read()
                assert open(fname, 'r').read() == base_amb_log

            estimate(respy_obj)

            if base_est_info is None:
                base_est_info = open('est.respy.info', 'r').read()
                assert open('est.respy.info', 'r').read() == base_est_info

            if base_est_log is None:
                base_est_log = open('est.respy.log', 'r').readlines()
            compare_est_log(base_est_log)
