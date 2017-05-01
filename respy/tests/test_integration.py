from pandas.util.testing import assert_frame_equal
import numpy as np
import pandas as pd
import pytest

from respy.python.shared.shared_auxiliary import print_init_dict
from respy.scripts.scripts_estimate import scripts_estimate
from respy.scripts.scripts_simulate import scripts_simulate
from respy.scripts.scripts_update import scripts_update
from respy.scripts.scripts_modify import scripts_modify
from respy.python.process.process_python import process
from codes.auxiliary import write_interpolation_grid
from codes.random_init import generate_random_dict
from respy.custom_exceptions import UserError
from codes.auxiliary import simulate_observed
from codes.random_init import generate_init

from respy import estimate
from respy import RespyCls


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ Testing whether random model specifications can be simulated and processed.
        """
        # Generate random initialization file
        generate_init()

        respy_obj = RespyCls('test.respy.ini')

        simulate_observed(respy_obj)

        process(respy_obj)

    def test_2(self):
        """ If there is no random variation in rewards then the number of draws to simulate the 
        expected future value should have no effect.
        """
        # Generate constraints
        constr = dict()
        constr['flag_deterministic'] = True

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

            respy_obj = simulate_observed(respy_obj)

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
        constr['flag_myopic'] = True

        generate_init(constr)

        # Iterate over alternative discount rates.
        base_data, base_val = None, None

        for delta in [0.00, 0.000001]:

            respy_obj = RespyCls('test.respy.ini')

            respy_obj.unlock()

            respy_obj.attr['optim_paras']['delta'] = np.array([delta])

            respy_obj.lock()

            simulate_observed(respy_obj)

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
        """ Test the evaluation of the criterion function for random requests, not just at the 
        true values.
        """
        # Constraints that ensure that two alternative initialization files can be used for the
        # same simulated data.
        constr = dict()
        constr['agents'] = np.random.randint(1, 100)
        constr['periods'] = np.random.randint(1, 4)
        constr['flag_ambiguity'] = flag_ambiguity
        constr['edu'] = (7, 15)
        constr['maxfun'] = 0

        # Simulate a dataset
        generate_init(constr)
        respy_obj = RespyCls('test.respy.ini')
        simulate_observed(respy_obj)

        # Evaluate at different points, ensuring that the simulated dataset still fits.
        generate_init(constr)

        respy_obj = RespyCls('test.respy.ini')
        estimate(respy_obj)

    def test_5(self):
        """ Test the scripts.
        """
        # Constraints that ensure that two alternative initialization files can be used for the
        # same simulated data.
        for _ in range(10):
            constr = dict()
            constr['periods'] = np.random.randint(1, 4)
            constr['agents'] = np.random.randint(5, 100)
            constr['flag_estimation'] = True
            constr['edu'] = (7, 15)

            # Simulate a dataset
            generate_init(constr)
            respy_obj = RespyCls('test.respy.ini')
            simulate_observed(respy_obj)

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

            # TODO: Extend set of admissible actions. If we want to include values and bounds
            # then we need to put in some extra effort to ensure that all are valid, i.e. that a
            # random value is within the bounds.
            action = np.random.choice(['fix', 'free'])
            num_draws = np.random.randint(1, 22)

            # The set of identifiers is a little complicated as we only allow sampling of the
            # diagonal terms of the covariance matrix. Otherwise, we sometimes run into the
            # problem of very ill conditioned matrices resulting in a failed Cholesky decomposition.
            set_ = list(range(22))

            identifiers = np.random.choice(set_, num_draws, replace=False)
            values = np.random.uniform(size=num_draws)

            scripts_estimate(single, init_file)
            scripts_update(init_file)

            # The error can occur as the RESPY package is actually running an estimation step
            # that can result in very ill-conditioned covariance matrices.
            try:
                scripts_simulate(init_file, file_sim)
                scripts_modify(identifiers, action, init_file, values)
            except np.linalg.linalg.LinAlgError:
                pass

    @pytest.mark.slow
    def test_6(self, flag_ambiguity=False):
        """ Test short estimation tasks.
        """
        constr = dict()
        constr['flag_estimation'] = True
        constr['flag_ambiguity'] = flag_ambiguity

        generate_init(constr)

        write_interpolation_grid('test.respy.ini')

        # Run estimation task.
        respy_obj = RespyCls('test.respy.ini')
        simulate_observed(respy_obj)
        base_x, base_val = estimate(respy_obj)

        # We also check whether updating the class instance and a single
        # evaluation of the criterion function give the same result.
        respy_obj.update_optim_paras(base_x)
        respy_obj.attr['maxfun'] = 0

        alt_x, alt_val = estimate(respy_obj)

        for arg in [(alt_val, base_val), (alt_x, base_x)]:
            np.testing.assert_almost_equal(arg[0], arg[1])

    def test_7(self, flag_ambiguity=False):
        """ We test whether a restart does result in the exact function
        evaluation. Additionally, we change the status of parameters at random.
        """
        constr = dict()
        constr['flag_estimation'] = True
        constr['flag_ambiguity'] = flag_ambiguity
        constr['version'] = 'PYTHON'
        generate_init(constr)

        respy_obj = RespyCls('test.respy.ini')
        respy_obj = simulate_observed(respy_obj)
        _, base_val = estimate(respy_obj)

        scripts_update('test.respy.ini')
        respy_obj = RespyCls('test.respy.ini')

        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)
        respy_obj.lock()

        action = np.random.choice(['fix', 'free'])
        num_draws = np.random.randint(1, 18)
        identifiers = np.random.choice(range(18), num_draws, replace=False)
        scripts_modify(identifiers, action, 'test.respy.ini')

        _, update_val = estimate(respy_obj)

        np.testing.assert_almost_equal(update_val, base_val)

    def test_8(self):
        """ This test ensures that the constraints for the covariance matrix are properly handled.
        """

        init_dict = generate_random_dict()

        # Manual specification of update patterns.
        updates = dict()

        updates['valid_1'] = [False, True, True, True, False, True, True, False, True, False]
        updates['valid_2'] = [False] * 10
        updates['valid_3'] = [True] * 10

        updates['invalid_1'] = [False, False, True, True, False, True, True, False, True, False]
        updates['invalid_2'] = [False, False, False, True, False, False, True, False, True, False]

        # We draw a random update and print it out to the initialization file.
        label = np.random.choice(list(updates.keys()))
        init_dict['SHOCKS']['fixed'] = updates[label]
        print_init_dict(init_dict)

        if 'invalid' in label:
            # This exception block makes sure that the UserError is in fact raised.
            try:
                RespyCls('test.respy.ini')
                raise AssertionError
            except UserError:
                pass
        else:
            RespyCls('test.respy.ini')

    def test_9(self):
        """ We ensure that the number of types does not matter for the evaluation of the criterion 
        function if a weight of one is put on the
        first group.
        """

        constr = dict()
        constr['flag_estimation'] = True

        # The interpolation equation is affected by the number of types
        # regardless of the weights.
        constr['flag_interpolation'] = False

        init_dict = generate_init(constr)

        base_val = None

        for num_types in [1, np.random.choice([2, 3, 4]).tolist()]:

            # We always need to ensure that a weight of one is on the first type. We need to fix
            # the weight in this case to not change during an estimation as well.
            shifts = np.random.uniform(-0.05, 0.05, size=(num_types - 1) * 4)
            shares = [1.0] + [0.0] * (num_types - 1)
            init_dict['TYPE_SHIFTS']['coeffs'] = shifts
            init_dict['TYPE_SHIFTS']['fixed'] = [False] * (num_types * 4)
            init_dict['TYPE_SHIFTS']['bounds'] = [[None, None]] * (num_types * 4)

            init_dict['TYPE_SHARES']['coeffs'] = shares
            init_dict['TYPE_SHARES']['fixed'] = [True] * num_types
            init_dict['TYPE_SHARES']['bounds'] = [[None, None]] * num_types
            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')
            simulate_observed(respy_obj)
            _, val = estimate(respy_obj)
            if base_val is None:
                base_val = val

            np.testing.assert_almost_equal(base_val, val)
