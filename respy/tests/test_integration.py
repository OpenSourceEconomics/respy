# standard library
import shutil

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

# testing library
from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_random_dict
from codes.random_init import generate_init
from codes.auxiliary import write_draws

# project library
from respy.scripts.scripts_estimate import scripts_estimate
from respy.scripts.scripts_simulate import scripts_simulate
from respy.scripts.scripts_update import scripts_update
from respy.scripts.scripts_modify import scripts_modify

from respy.python.shared.shared_auxiliary import print_init_dict

from respy.python.solve.solve_auxiliary import pyth_create_state_space


from respy.python.process.process_python import process

from respy.solve import solve
from respy import estimate
from respy import simulate
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ Testing whether random model specifications can be solved, simulated
        and processed.
        """
        # Generate random initialization file
        generate_init()

        respy_obj = RespyCls('test.respy.ini')

        solve(respy_obj)

        simulate(respy_obj)

        process(respy_obj)

    def test_2(self):
        """ Testing the equality of an evaluation of the criterion function for
        a random request.
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
            respy_obj = solve(respy_obj)
            simulate(respy_obj)

            # This parts checks the equality of simulated dataset for the
            # different versions of the code.
            data_frame = pd.read_csv('data.respy.dat', delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the
            # criterion function.
            simulate(respy_obj)

            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-05,
                                       atol=1e-06)

            # We know even more for the deterministic case.
            if constr['is_deterministic']:
                assert (crit_val in [0.0, 1.0])

    def test_3(self):
        """ If there is no random variation in payoffs then the number of
        draws to simulate the expected future value should have no effect.
        """
        # Generate constraints
        constr = dict()
        constr['is_deterministic'] = True

        # Generate random initialization file
        generate_init(constr)

        # Initialize auxiliary objects
        base = None

        for _ in range(2):

            # Draw a random number of draws for
            # expected future value calculations.
            num_draws_emax = np.random.randint(1, 100)

            # Perform toolbox actions
            respy_obj = RespyCls('test.respy.ini')

            respy_obj.unlock()

            respy_obj.set_attr('num_draws_emax', num_draws_emax)

            respy_obj.lock()

            respy_obj = solve(respy_obj)

            # Distribute class attributes
            periods_emax = respy_obj.get_attr('periods_emax')

            if base is None:
                base = periods_emax.copy()

            # Statistic
            diff = np.max(abs(np.ma.masked_invalid(base) - np.ma.masked_invalid(
                periods_emax)))

            # Checks
            assert (np.isfinite(diff))
            assert (diff < 10e-10)

    def test_4(self):
        """ Testing whether the a simulated dataset and the evaluation of the
        criterion function are the same for a tiny delta and a myopic agent.
        """

        # Generate random initialization dictionary
        constr = dict()
        constr['maxfun'] = 0

        generate_init(constr)

        # Iterate over alternative discount rates.
        base_data, base_val = None, None

        for delta in [0.00, 0.000001]:

            respy_obj = RespyCls('test.respy.ini')

            respy_obj.unlock()

            respy_obj.set_attr('delta', delta)

            respy_obj.lock()

            solve(respy_obj)

            simulate(respy_obj)

            # This parts checks the equality of simulated dataset for the
            # different versions of the code.
            data_frame = pd.read_csv('data.respy.dat', delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the
            # criterion function.
            simulate(respy_obj)

            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-03, atol=1e-03)

    def test_5(self):
        """ This test ensures that the evaluation of the criterion function
        at the starting value is identical between the different versions.
        """

        max_draws = np.random.randint(10, 100)

        # Generate random initialization file
        constr = dict()
        constr['flag_parallelism'] = False
        constr['max_draws'] = max_draws
        constr['flag_interpolation'] = False
        constr['maxfun'] = 0

        # Generate random initialization file
        init_dict = generate_init(constr)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # Simulate a dataset
        simulate(respy_obj)

        # Iterate over alternative implementations
        base_x, base_val, base_log = None, None, None

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

            solve(respy_obj)

            # Check for identical logging
            if base_log is None:
                base_log = open('sol.respy.log', 'r').read()
            assert open('sol.respy.log', 'r').read() == base_log

    def test_6(self):
        """ Test the evaluation of the criterion function for random
        requests, not just at the true values.
        """
        # Constraints that ensure that two alternative initialization files
        # can be used for the same simulated data.
        constr = dict()
        constr['periods'] = np.random.randint(1, 4)
        constr['agents'] = np.random.randint(1, 100)
        constr['edu'] = (7, 15)
        constr['maxfun'] = 0

        # Simulate a dataset
        generate_init(constr)
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)

        # Evaluate at different points, ensuring that the simulated dataset
        # still fits.
        generate_init(constr)

        respy_obj = RespyCls('test.respy.ini')
        estimate(respy_obj)

    def test_7(self):
        """ Test the scripts.
        """
        # Constraints that ensure that two alternative initialization files
        # can be used for the same simulated data.
        constr = dict()
        constr['periods'] = np.random.randint(1, 4)
        constr['agents'] = np.random.randint(5, 100)
        constr['is_estimation'] = True
        constr['edu'] = (7, 15)

        # Simulate a dataset
        generate_init(constr)
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)

        # Potentially evaluate at different points.
        generate_init(constr)
        shutil.move('data.respy.paras', 'est.respy.step')

        init_file = 'test.respy.ini'
        file_sim = 'sim.respy'

        gradient = np.random.choice([True, False])
        single = np.random.choice([True, False])
        resume = np.random.choice([True, False])
        update = np.random.choice([True, False])

        action = np.random.choice(['fix', 'free', 'value'])
        num_draws = np.random.randint(1, 20)

        # The set of identifiers is a little complicated as we only allow
        # sampling of the diagonal terms of the covariance matrix. Otherwise,
        # we sometimes run into the problem of very ill conditioned matrices
        # resulting in a failed Cholesky decomposition.
        set_ = list(range(16)) + [16, 18, 21, 25]

        identifiers = np.random.choice(set_, num_draws, replace=False)
        values = np.random.uniform(size=num_draws)

        scripts_estimate(resume, single, init_file, gradient)
        scripts_update(init_file)

        # The error can occur as the RESPY package is actually running an
        # estimation step that can result in very ill-conditioned covariance
        # matrices.
        try:
            scripts_simulate(update, init_file, file_sim, None)
            scripts_modify(identifiers, values, action, init_file)
        except np.linalg.linalg.LinAlgError:
            pass

    @pytest.mark.slow
    def test_8(self):
        """ Test short estimation tasks.
        """
        # Constraints that ensures that the maximum number of iterations and
        # the number of function evaluations is set to the minimum values of
        # one.
        constr = dict()
        constr['is_estimation'] = True

        generate_init(constr)

        # Run estimation task.
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)
        estimate(respy_obj)
