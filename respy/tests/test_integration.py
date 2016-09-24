from pandas.util.testing import assert_frame_equal
import numpy as np
import pandas as pd
import pytest

from respy.scripts.scripts_estimate import scripts_estimate
from respy.scripts.scripts_simulate import scripts_simulate
from respy.scripts.scripts_update import scripts_update
from respy.scripts.scripts_modify import scripts_modify
from respy.python.process.process_python import process
from codes.random_init import generate_init
from respy import estimate
from respy import simulate
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ Testing whether random model specifications can be simulated
        and processed.
        """
        # Generate random initialization file
        generate_init()

        respy_obj = RespyCls('test.respy.ini')

        simulate(respy_obj)

        process(respy_obj)

    def test_2(self):
        """ If there is no random variation in rewards then the number of
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

            respy_obj = simulate(respy_obj)

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

    def test_3(self, flag_ambiguity=False):
        """ Testing whether the a simulated dataset and the evaluation of the
        criterion function are the same for a tiny delta and a myopic agent.
        """

        # Generate random initialization dictionary
        constr = dict()
        constr['maxfun'] = 0
        constr['flag_ambiguity'] = flag_ambiguity

        generate_init(constr)

        # Iterate over alternative discount rates.
        base_data, base_val = None, None

        for delta in [0.00, 0.000001]:

            respy_obj = RespyCls('test.respy.ini')

            respy_obj.unlock()

            respy_obj.set_attr('delta', delta)

            respy_obj.lock()

            simulate(respy_obj)

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

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-03, atol=1e-03)

    def test_4(self, flag_ambiguity=False):
        """ Test the evaluation of the criterion function for random
        requests, not just at the true values.
        """
        # Constraints that ensure that two alternative initialization files
        # can be used for the same simulated data.
        constr = dict()
        constr['agents'] = np.random.randint(1, 100)
        constr['periods'] = np.random.randint(1, 4)
        constr['flag_ambiguity'] = flag_ambiguity
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

    def test_5(self):
        """ Test the scripts.
        """
        # Constraints that ensure that two alternative initialization files
        # can be used for the same simulated data.
        for _ in range(10):
            constr = dict()
            constr['periods'] = np.random.randint(1, 4)
            constr['agents'] = np.random.randint(5, 100)
            constr['is_estimation'] = True
            constr['edu'] = (7, 15)

            # Simulate a dataset
            generate_init(constr)
            respy_obj = RespyCls('test.respy.ini')
            simulate(respy_obj)

            # Create output to process a baseline.
            respy_obj.unlock()
            respy_obj.set_attr('maxfun', 0)
            respy_obj.lock()

            estimate(respy_obj)

            # Potentially evaluate at different points.
            generate_init(constr)

            init_file = 'test.respy.ini'
            file_sim = 'sim.respy.dat'

            single = np.random.choice([True, False])

            action = np.random.choice(['fix', 'free', 'value'])
            num_draws = np.random.randint(1, 20)

            # The set of identifiers is a little complicated as we only allow
            # sampling of the diagonal terms of the covariance matrix.
            # Otherwise, we sometimes run into the problem of very ill
            # conditioned matrices resulting in a failed Cholesky decomposition.
            set_ = list(range(17)) + [17, 19, 22, 26]

            identifiers = np.random.choice(set_, num_draws, replace=False)
            values = np.random.uniform(size=num_draws)

            scripts_estimate(single, init_file)
            scripts_update(init_file)

            # The error can occur as the RESPY package is actually running an
            # estimation step that can result in very ill-conditioned covariance
            # matrices.
            try:
                scripts_simulate(init_file, file_sim)
                scripts_modify(identifiers, action, init_file, values)
            except np.linalg.linalg.LinAlgError:
                pass

    @pytest.mark.slow
    def test_6(self, flag_ambiguity=False):
        """ Test short estimation tasks.
        """
        # Constraints that ensures that the maximum number of iterations and
        # the number of function evaluations is set to the minimum values of
        # one.
        constr = dict()
        constr['is_estimation'] = True
        constr['flag_ambiguity'] = flag_ambiguity

        generate_init(constr)

        # Run estimation task.
        respy_obj = RespyCls('test.respy.ini')
        simulate(respy_obj)
        estimate(respy_obj)

    def test_7(self, flag_ambiguity=False):
        """ We test whether a restart does result in the exact function
        evaluation. Additionally, we change the status of parameters at random.
        """
        constr = dict()
        constr['is_estimation'] = True
        constr['flag_ambiguity'] = flag_ambiguity
        generate_init(constr)

        respy_obj = RespyCls('test.respy.ini')
        respy_obj = simulate(respy_obj)
        _, base_val = estimate(respy_obj)

        scripts_update('test.respy.ini')

        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)
        respy_obj.lock()

        action = np.random.choice(['fix', 'free'])
        num_draws = np.random.randint(1, 27)
        identifiers = np.random.choice(range(27), num_draws, replace=False)
        scripts_modify(identifiers, action, 'test.respy.ini')

        _, update_val = estimate(respy_obj)

        np.testing.assert_almost_equal(update_val, base_val)
