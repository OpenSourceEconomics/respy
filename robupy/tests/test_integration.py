""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal

import numpy as np
import pandas as pd

import shutil
import pytest

# testing library
from codes.auxiliary import write_interpolation_grid
from codes.auxiliary import write_draws

# project library
from robupy.scripts.scripts_estimate import scripts_estimate
from robupy.scripts.scripts_simulate import scripts_simulate
from robupy.scripts.scripts_update import scripts_update
from robupy.scripts.scripts_modify import scripts_modify

from robupy.python.shared.shared_auxiliary import dist_class_attributes
from robupy.python.shared.shared_auxiliary import dist_model_paras
from robupy.python.shared.shared_auxiliary import read_draws

from robupy.python.solve.solve_auxiliary import pyth_create_state_space

from robupy.python.estimate.estimate_auxiliary import get_optim_paras
from robupy.tests.codes.random_init import generate_random_dict
from robupy.tests.codes.random_init import print_random_dict
from robupy.tests.codes.random_init import generate_init

from robupy import simulate
from robupy import evaluate
from robupy import estimate
from robupy import process
from robupy import solve
from robupy import read

from robupy.python.solve.solve_python import pyth_solve
from robupy.fortran.f2py_library import f2py_solve
from robupy.fortran.fortran import fort_solve

from robupy.python.simulate.simulate_python import pyth_simulate
from robupy.fortran.f2py_library import f2py_simulate

from robupy.python.evaluate.evaluate_python import pyth_evaluate
from robupy.fortran.f2py_library import f2py_evaluate
from robupy.fortran.fortran import fort_evaluate

from robupy.python.estimate.estimate_python import pyth_criterion
from robupy.fortran.f2py_library import f2py_criterion

''' Main
'''


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

        robupy_obj = read('test.robupy.ini')

        solve(robupy_obj)

        simulate(robupy_obj)

        process(robupy_obj)

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
        constraints = dict()
        constraints['is_deterministic'] = is_deterministic
        constraints['is_myopic'] = is_myopic
        constraints['max_draws'] = max_draws

        init_dict = generate_random_dict(constraints)

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
            init_dict['INTERPOLATION']['apply'] = True
            init_dict['INTERPOLATION']['points'] = \
                np.random.randint(10, max_states_period)

        # Print out the relevant initialization file.
        print_random_dict(init_dict)

        # Write out random components and interpolation grid to align the
        # three implementations.
        num_periods = init_dict['BASICS']['periods']
        write_draws(num_periods, max_draws)
        write_interpolation_grid('test.robupy.ini')

        # Clean evaluations based on interpolation grid,
        base_val, base_data = None, None

        for version in ['PYTHON', 'F2PY', 'FORTRAN']:
            robupy_obj = read('test.robupy.ini')

            # Modify the version of the program for the different requests.
            robupy_obj.unlock()
            robupy_obj.set_attr('version',  version)
            robupy_obj.lock()

            # Solve the model
            robupy_obj = solve(robupy_obj)
            simulate(robupy_obj)

            # This parts checks the equality of simulated dataset for the
            # different versions of the code.
            data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the
            # criterion function.
            data_frame, _ = simulate(robupy_obj)

            crit_val = evaluate(robupy_obj, data_frame)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-05,
                                       atol=1e-06)

            # We know even more for the deterministic case.
            if constraints['is_deterministic']:
                assert (crit_val in [0.0, 1.0])

    def test_3(self):
        """ If there is no random variation in payoffs then the number of
        draws to simulate the expected future value should have no effect.
        """
        # Generate constraints
        constraints = dict()
        constraints['is_deterministic'] = True
        constraints['level'] = 0.0

        # Generate random initialization file
        generate_init(constraints)

        # Initialize auxiliary objects
        base = None

        for _ in range(2):

            # Draw a random number of draws for
            # expected future value calculations.
            num_draws_emax = np.random.randint(1, 100)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')

            robupy_obj.unlock()

            robupy_obj.set_attr('num_draws_emax', num_draws_emax)

            robupy_obj.lock()

            robupy_obj = solve(robupy_obj)

            # Distribute class attributes
            periods_emax = robupy_obj.get_attr('periods_emax')

            if base is None:
                base = periods_emax.copy()

            # Statistic
            diff = np.max(abs(np.ma.masked_invalid(base) - np.ma.masked_invalid(
                periods_emax)))

            # Checks
            assert (np.isfinite(diff))
            assert (diff < 10e-10)

    def test_4(self):
        """ Testing whether the risk code is identical to the ambiguity code for
        very, very small levels of ambiguity.
        """
        # Generate random initialization dictionary
        constraints = dict()

        init_dict = generate_random_dict(constraints)

        # Initialize containers
        base = None

        # Loop over different uncertain environments.
        for level in [0.00, 0.000000000000001]:

            # Set varying constraints
            init_dict['AMBIGUITY']['level'] = level

            # Print to dictionary
            print_random_dict(init_dict)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')

            robupy_obj = solve(robupy_obj)

            # Distribute class attributes
            periods_emax = robupy_obj.get_attr('periods_emax')

            if base is None:
                base = periods_emax.copy()

            # Checks
            np.testing.assert_allclose(base, periods_emax, rtol=1e-06)

    def test_5(self):
        """ Testing whether the systematic payoff calculation is unaffected by
        the level of ambiguity.
        """
        # Select version

        # Generate constraints
        constraints = dict()

        # Generate random initialization dictionary
        init_dict = generate_random_dict(constraints)

        # Initialize containers
        base = None

        # Loop over different uncertain environments.
        for _ in range(2):

            # Set varying constraints
            init_dict['AMBIGUITY']['level'] = np.random.choice(
                [0.00, np.random.uniform()])

            # Print to dictionary
            print_random_dict(init_dict)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')

            robupy_obj = solve(robupy_obj)

            # Distribute class attributes
            systematic = robupy_obj.get_attr('periods_payoffs_systematic')

            if base is None:
                base = systematic.copy()

            # Checks
            np.testing.assert_allclose(base, systematic)

    def test_6(self):
        """ Testing whether the a simulated dataset and the evaluation of the
        criterion function are the same for a tiny delta and a myopic agent.
        """

        # Generate random initialization dictionary
        generate_init()

        # Iterate over alternative discount rates.
        base_data, base_val = None, None

        for delta in [0.00, 0.000001]:

            robupy_obj = read('test.robupy.ini')

            robupy_obj.unlock()

            robupy_obj.set_attr('delta', delta)

            robupy_obj.lock()

            solve(robupy_obj)

            simulate(robupy_obj)

            # This parts checks the equality of simulated dataset for the
            # different versions of the code.
            data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

            if base_data is None:
                base_data = data_frame.copy()

            assert_frame_equal(base_data, data_frame)

            # This part checks the equality of an evaluation of the
            # criterion function.
            data_frame, _ = simulate(robupy_obj)

            crit_val = evaluate(robupy_obj, data_frame)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val, rtol=1e-03, atol=1e-03)

    def test_7(self):
        """ This methods ensures that the core functions yield the same
        results across implementations.
        """

        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        max_states_period = write_interpolation_grid('test.robupy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, model_paras, num_draws_emax, \
            seed_emax, is_debug, delta, level, is_ambiguous, \
            is_interpolated, num_points, is_deterministic, is_myopic, \
            num_agents_sim, num_draws_prob, seed_prob, tau, paras_fixed = \
                dist_class_attributes(robupy_obj,
                    'num_periods', 'edu_start', 'edu_max', 'min_idx',
                    'model_paras', 'num_draws_emax', 'seed_emax', 'is_debug',
                    'delta', 'level', 'is_ambiguous',
                    'is_interpolated', 'num_points', 'is_deterministic',
                    'is_myopic', 'num_agents_sim', 'num_draws_prob',
                    'seed_prob', 'tau', 'paras_fixed')

        # Write out random components and interpolation grid to align the
        # three implementations.
        max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
        write_draws(num_periods, max_draws)
        periods_draws_emax = read_draws(num_periods, num_draws_emax)
        periods_draws_prob = read_draws(num_periods, num_draws_prob)
        periods_draws_sims = read_draws(num_periods, num_agents_sim)

        # Extract coefficients
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, \
            shocks_cholesky = dist_model_paras(model_paras, True)

        # Check the full solution procedure
        base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, is_debug,
            edu_max, min_idx, delta, level)

        fort = fort_solve(*base_args + (seed_emax, tau))
        pyth = pyth_solve(*base_args + (periods_draws_emax,))
        f2py = f2py_solve(*base_args + (periods_draws_emax, max_states_period))

        for alt in [f2py, fort]:
            for i in range(5):
                np.testing.assert_allclose(pyth[i], alt[i])

        # Distribute solution arguments for further use in simulation test.
        periods_payoffs_systematic, states_number_period, mapping_state_idx, \
            periods_emax, states_all = pyth

        # Collect arguments across implementations.
        args = (periods_payoffs_systematic, mapping_state_idx, periods_emax,
            num_periods, states_all, num_agents_sim, edu_start, edu_max, delta,
            periods_draws_sims, shocks_cholesky)

        pyth = pyth_simulate(*args)
        f2py = f2py_simulate(*args)

        np.testing.assert_allclose(pyth, f2py)

        data_array = pyth

        base_args = (coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov,
            is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, is_debug,
            edu_max, min_idx, delta, level, data_array, num_agents_sim,
            num_draws_prob, tau)

        args = base_args + (seed_emax, seed_prob)
        fort = fort_evaluate(*args)

        args = base_args + (periods_draws_emax, periods_draws_prob)
        pyth = pyth_evaluate(*args)

        args = base_args + (periods_draws_emax, periods_draws_prob)
        f2py = f2py_evaluate(*args)

        for alt in [f2py, fort]:
            np.testing.assert_allclose(pyth, alt)

        # Evaluation of criterion function
        x0 = get_optim_paras(coeffs_a, coeffs_b, coeffs_edu,
                coeffs_home, shocks_cov, 'all', paras_fixed, is_debug)

        args = (is_deterministic, is_interpolated, num_draws_emax, is_ambiguous,
            num_periods, num_points, is_myopic, edu_start, is_debug,
            edu_max, min_idx, delta, level, data_array, num_agents_sim,
            num_draws_prob, tau, periods_draws_emax, periods_draws_prob)

        pyth = pyth_criterion(x0, *args)
        f2py = f2py_criterion(x0, *args)
        np.testing.assert_allclose(pyth, f2py)

    def test_8(self):
        """ This test ensures that the evaluation of the criterion function
        at the starting value is identical between the different versions.
        """

        # Generate random initialization file
        constraints = dict()
        constraints['maxiter'] = 0

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        # Simulate a dataset
        simulate(robupy_obj)

        # Iterate over alternative implementations
        base_x, base_val = None, None

        for version in ['FORTRAN', 'PYTHON', 'F2PY']:

            robupy_obj.unlock()

            robupy_obj.set_attr('version', version)

            robupy_obj.lock()

            x, val = estimate(robupy_obj, process(robupy_obj))

            # Check for the returned parameters.
            if base_x is None:
                base_x = x
            np.testing.assert_allclose(base_x, x)

            # Check for the value of the criterion function.
            if base_val is None:
                base_val = val
            np.testing.assert_allclose(base_val, val)

    def test_9(self):
        """ Test the evaluation of the criterion function for random
        requests, not just at the true values.
        """
        # Constraints that ensure that two alternative initialization files
        # can be used for the same simulated data.
        constr = dict()
        constr['periods'] = np.random.randint(1, 4)
        constr['agents'] = np.random.randint(1, 100)
        constr['edu'] = (7, 15)
        constr['maxiter'] = 0

        # Simulate a dataset
        generate_init(constr)
        robupy_obj = read('test.robupy.ini')
        data_frame, _ = simulate(robupy_obj)

        # Evaluate at different points, ensuring that the simulated datset
        # still fits.
        generate_init(constr)

        robupy_obj = read('test.robupy.ini')
        evaluate(robupy_obj, data_frame)

    def test_10(self):
        """ Test the scripts.
        """
        # Constraints that ensure that two alternative initialization files
        # can be used for the same simulated data.
        constr = dict()
        constr['periods'] = np.random.randint(1, 4)
        constr['agents'] = np.random.randint(5, 100)
        constr['edu'] = (7, 15)
        constr['maxiter'] = 0

        # Simulate a dataset
        generate_init(constr)
        robupy_obj = read('test.robupy.ini')
        simulate(robupy_obj)

        # Potentially evaluate at different points.
        generate_init(constr)
        shutil.move('data.robupy.paras', 'paras_steps.robupy.log')

        init_file = 'test.robupy.ini'
        file_sim = 'sim.robupy'

        gradient = np.random.choice([True, False])
        single = np.random.choice([True, False])
        resume = np.random.choice([True, False])
        update = np.random.choice([True, False])

        action = np.random.choice(['fix', 'free', 'value'])
        num_draws = np.random.randint(1, 27)
        identifiers = np.random.choice(range(26), num_draws, replace=False)
        values = np.random.uniform(size=num_draws)

        scripts_estimate(resume, single, init_file, gradient)
        scripts_simulate(update, init_file, file_sim, None)
        scripts_update(init_file)
        scripts_modify(identifiers, values, action, init_file)

    def test_11(self):
        """ Test short estimation tasks.
        """
        # Constraints that ensures that the maximum number of iterations and
        # the number of function evaluations is set to the minimum values of
        # one.
        constr = dict()
        constr['is_estimation'] = True
        constr['level'] = 0.0

        generate_init(constr)

        # Run estimation task.
        robupy_obj = read('test.robupy.ini')
        data_frame, _ = simulate(robupy_obj)
        estimate(robupy_obj, data_frame)
