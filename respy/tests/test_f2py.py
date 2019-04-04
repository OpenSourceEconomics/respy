import sys
import os
import statsmodels.api as sm
from scipy.stats import norm
import numpy as np
import pytest
import scipy
from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.record.record_estimation import _spectral_condition_number
from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_auxiliary import transform_disturbances
from respy.python.solve.solve_auxiliary import StateSpace
from respy.python.solve.solve_auxiliary import pyth_backward_induction
from respy.python.solve.solve_auxiliary import get_simulated_indicator
from respy.python.solve.solve_auxiliary import get_exogenous_variables
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.solve.solve_auxiliary import get_endogenous_variable
from respy.python.shared.shared_auxiliary import get_emaxs_of_subsequent_period
from respy.python.shared.shared_constants import TEST_RESOURCES_BUILD
from respy.python.evaluate.evaluate_python import pyth_contributions
from respy.python.simulate.simulate_auxiliary import sort_type_info
from respy.python.simulate.simulate_auxiliary import sort_edu_spec
from respy.python.shared.shared_auxiliary import extract_cholesky
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.estimate.estimate_python import pyth_criterion
from respy.python.simulate.simulate_python import pyth_simulate
from respy.python.solve.solve_auxiliary import get_predictions
from respy.python.solve.solve_risk import construct_emax_risk
from respy.python.shared.shared_auxiliary import create_draws
from respy.python.shared.shared_auxiliary import read_draws
from respy.python.shared.shared_constants import IS_F2PY
from respy.python.interface import get_scales_magnitudes
from respy.pre_processing.data_processing import process_dataset
from respy.python.solve.solve_python import pyth_solve
from respy.fortran.interface import resfort_interface
from respy.tests.codes.auxiliary import write_interpolation_grid
from respy.tests.codes.auxiliary import write_lagged_start
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model
from respy.tests.codes.auxiliary import write_edu_start
from respy.tests.codes.auxiliary import write_draws
from respy.tests.codes.auxiliary import write_types
from functools import partial
from numpy.testing import (
    assert_equal,
    assert_array_equal,
    assert_array_almost_equal,
)
from respy import RespyCls

from respy.python.shared.shared_constants import DECIMALS, TOL, MISSING_FLOAT

assert_allclose = partial(np.testing.assert_allclose, rtol=TOL, atol=TOL)
assert_almost_equal = partial(np.testing.assert_almost_equal, decimal=DECIMALS)

if IS_F2PY:
    print(TEST_RESOURCES_BUILD)
    print(sys.path)
    print(os.listdir(str(TEST_RESOURCES_BUILD)))
    sys.path.insert(0, str(TEST_RESOURCES_BUILD))
    import f2py_interface as fort_debug


@pytest.mark.skipif(not IS_F2PY, reason="No F2PY available")
class TestClass(object):
    """ This class groups together some tests.
    """

    def test_1(self):
        """ Compare the evaluation of the criterion function for the ambiguity
        optimization and the simulated expected future value between the FORTRAN and
        PYTHON implementations. These tests are set up a separate test case due to the
        large setup cost to construct the ingredients for the interface.
        """
        # Generate constraint periods
        constr = dict()
        constr["version"] = "python"
        # Generate random initialization file
        params_spec, options_spec = generate_random_model(point_constr=constr)
        respy_obj = RespyCls(params_spec, options_spec)
        respy_obj = simulate_observed(respy_obj)

        # Extract class attributes
        (
            state_space,
            states_all,
            mapping_state_idx,
            periods_rewards_systematic,
            periods_emax,
            num_periods,
            num_draws_emax,
            edu_spec,
            optim_paras,
            num_types,
        ) = dist_class_attributes(
            respy_obj,
            "state_space",
            "states_all",
            "mapping_state_idx",
            "periods_rewards_systematic",
            "periods_emax",
            "num_periods",
            "num_draws_emax",
            "edu_spec",
            "optim_paras",
            "num_types",
        )

        # Sample draws
        draws_emax_standard = np.random.multivariate_normal(
            np.zeros(4), np.identity(4), num_draws_emax
        )
        draws_emax_risk = transform_disturbances(
            draws_emax_standard, np.zeros(4), optim_paras["shocks_cholesky"]
        )

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(state_space.states_per_period[period]))

        # Select systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        # Evaluation of simulated expected future values. Limit to one individual as the
        # Fortran version.
        rewards_period = state_space.get_attribute_from_period(
            "rewards", period
        )[k]
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[
            k, :4
        ]

        py = construct_emax_risk(
            rewards_period[-2:],
            rewards_period[:4],
            emaxs_period,
            draws_emax_risk,
            optim_paras["delta"],
        )

        f90 = fort_debug.wrapper_construct_emax_risk(
            num_periods,
            num_draws_emax,
            period,
            k,
            draws_emax_risk,
            rewards_systematic,
            periods_emax,
            states_all,
            mapping_state_idx,
            edu_spec["start"],
            edu_spec["max"],
            optim_paras["delta"],
            optim_paras["coeffs_common"],
            optim_paras["coeffs_a"],
            optim_paras["coeffs_b"],
            num_types,
        )

        assert_allclose(py, f90)

    def test_2(self):
        """ Compare results between FORTRAN and PYTHON of selected hand-crafted
        functions. In test_97() we test FORTRAN implementations against PYTHON intrinsic
        routines.
        """
        for _ in range(33):

            # Create grid of admissible state space values.
            num_edu_start = np.random.choice(range(1, 3))
            num_periods = np.random.randint(1, 15)
            num_types = np.random.randint(1, 3)

            edu_spec = dict()
            edu_spec["start"] = np.random.choice(
                range(1, 10), size=num_edu_start, replace=False
            ).tolist()
            edu_spec["max"] = max(edu_spec["start"]) + np.random.randint(1, 5)
            min_idx = edu_spec["max"] + 1

            # FORTRAN
            base_args = (num_periods, num_types)

            state_space = StateSpace(
                *base_args, edu_spec["start"], edu_spec["max"]
            )

            py_a, py_c, _, _ = state_space._get_fortran_counterparts()
            py_b = state_space.states_per_period
            py_d = py_b.max()

            fort_a, fort_b, fort_c, fort_d = fort_debug.wrapper_create_state_space(
                *base_args, edu_spec["start"], edu_spec["max"], min_idx
            )

            # Ensure equivalence
            rslts = [
                [fort_a, py_a],
                [fort_b, py_b],
                [fort_c, py_c],
                [fort_d, py_d],
            ]
            for obj in rslts:
                # Slice Fortran output to shape of Python output.
                if isinstance(obj[0], np.ndarray):
                    obj[0] = obj[0][tuple(map(slice, obj[1].shape))]

                assert_allclose(obj[0], obj[1])

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
            f90 = fort_debug.wrapper_get_coefficients(
                endog, exog, num_covars, num_agents
            )
            assert_almost_equal(py, f90)

            # Check prediction
            py = results.predict(exog)
            f90 = fort_debug.wrapper_point_predictions(exog, f90, num_agents)
            assert_almost_equal(py, f90)

            # Check coefficient of determination and the standard errors.
            py = [results.rsquared, results.bse]
            f90 = fort_debug.wrapper_get_pred_info(
                endog, f90, exog, num_agents, num_covars
            )

            for i in range(2):
                assert_almost_equal(py[i], f90[i])

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

            assert_almost_equal(py, f90)

            # Singular Value Decomposition
            py = scipy.linalg.svd(matrix)
            f90 = fort_debug.wrapper_svd(matrix, dim)

            for i in range(3):
                assert_allclose(py[i], f90[i])

            # Pseudo-Inverse
            py = np.linalg.pinv(matrix)
            f90 = fort_debug.wrapper_pinv(matrix, dim)

            assert_allclose(py, f90)

            # Inverse
            py = np.linalg.inv(cov)
            f90 = fort_debug.wrapper_inverse(cov, dim)
            assert_allclose(py, f90)

            # Determinant
            py = np.linalg.det(cov)
            f90 = fort_debug.wrapper_determinant(cov)

            assert_allclose(py, f90)

            # Trace
            py = np.trace(cov)
            f90 = fort_debug.wrapper_trace(cov)

            assert_allclose(py, f90)

            # Random normal deviates. This only tests the interface, requires
            # visual inspection in IPYTHON notebook as well.
            fort_debug.wrapper_standard_normal(num_draws_emax)

            # Clipping values below and above bounds.
            num_values = np.random.randint(1, 10000)
            lower_bound = np.random.randn()
            upper_bound = lower_bound + np.random.ranf()
            values = np.random.normal(size=num_values)

            f90 = fort_debug.wrapper_clip_value(
                values, lower_bound, upper_bound, num_values
            )
            py = np.clip(values, lower_bound, upper_bound)

            assert_almost_equal(py, f90)

            # Spectral condition number
            py = _spectral_condition_number(cov)
            fort = fort_debug.wrapper_spectral_condition_number(cov)
            assert_almost_equal(py, fort)

    def test_4(self):
        """ Testing the core functions of the solution step for the equality of results
        between the PYTHON and FORTRAN implementations.
        """
        params_spec, options_spec = generate_random_model()
        respy_obj = RespyCls(params_spec, options_spec)

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        write_interpolation_grid(respy_obj)

        # Extract class attributes
        (
            num_periods,
            edu_spec,
            optim_paras,
            num_draws_emax,
            seed_emax,
            is_debug,
            is_interpolated,
            num_points_interp,
            optimizer_options,
            file_sim,
            num_types,
        ) = dist_class_attributes(
            respy_obj,
            "num_periods",
            "edu_spec",
            "optim_paras",
            "num_draws_emax",
            "seed_emax",
            "is_debug",
            "is_interpolated",
            "num_points_interp",
            "optimizer_options",
            "file_sim",
            "num_types",
        )

        shocks_cholesky = optim_paras["shocks_cholesky"]
        coeffs_common = optim_paras["coeffs_common"]
        coeffs_home = optim_paras["coeffs_home"]
        coeffs_edu = optim_paras["coeffs_edu"]
        coeffs_a = optim_paras["coeffs_a"]
        coeffs_b = optim_paras["coeffs_b"]
        delta = optim_paras["delta"]

        type_spec_shifts = optim_paras["type_shifts"]
        type_spec_shares = optim_paras["type_shares"]

        min_idx = edu_spec["max"] + 1

        # Check the state space creation.
        state_space = StateSpace(
            num_periods,
            num_types,
            edu_spec["start"],
            edu_spec["max"],
            optim_paras,
        )

        states_all, mapping_state_idx, _, _ = (
            state_space._get_fortran_counterparts()
        )

        pyth = [
            states_all,
            state_space.states_per_period,
            mapping_state_idx,
            state_space.states_per_period.max(),
        ]

        f2py = fort_debug.wrapper_create_state_space(
            num_periods, num_types, edu_spec["start"], edu_spec["max"], min_idx
        )
        for i in range(4):
            # Slice Fortran output to shape of Python output.
            if isinstance(f2py[i], np.ndarray):
                f2py_reduced = f2py[i][tuple(map(slice, pyth[i].shape))]
            else:
                f2py_reduced = f2py[i]

            assert_allclose(pyth[i], f2py_reduced)

        _, _, pyth, _ = state_space._get_fortran_counterparts()

        args = (
            num_periods,
            state_space.states_per_period,
            states_all,
            state_space.states_per_period.max(),
            coeffs_common,
            coeffs_a,
            coeffs_b,
            coeffs_edu,
            coeffs_home,
            type_spec_shares,
            type_spec_shifts,
        )
        f2py = fort_debug.wrapper_calculate_rewards_systematic(*args)

        assert_allclose(pyth, f2py)

        # Carry some results from the systematic rewards calculation for future use and
        # create the required set of disturbances.
        periods_draws_emax = create_draws(
            num_periods, num_draws_emax, seed_emax, is_debug
        )

        # Save result for next test.
        periods_rewards_systematic = pyth.copy()

        # Check backward induction procedure.
        state_space = pyth_backward_induction(
            False,
            periods_draws_emax,
            state_space,
            is_debug,
            is_interpolated,
            num_points_interp,
            edu_spec,
            optim_paras,
            file_sim,
            False,
        )
        _, _, _, pyth = state_space._get_fortran_counterparts()

        args = (
            num_periods,
            False,
            state_space.states_per_period.max(),
            periods_draws_emax,
            num_draws_emax,
            state_space.states_per_period,
            periods_rewards_systematic,
            mapping_state_idx,
            states_all,
            is_debug,
            is_interpolated,
            num_points_interp,
            edu_spec["start"],
            edu_spec["max"],
            shocks_cholesky,
            delta,
            coeffs_common,
            coeffs_a,
            coeffs_b,
            file_sim,
            False,
        )
        f2py = fort_debug.wrapper_backward_induction(*args)

        assert_allclose(pyth, f2py)

    def test_5(self):
        """ This methods ensures that the core functions yield the same results across
        implementations.
        """
        params_spec, options_spec = generate_random_model()
        respy_obj = RespyCls(params_spec, options_spec)
        respy_obj = simulate_observed(respy_obj)

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        max_states_period = write_interpolation_grid(respy_obj)

        # Extract class attributes
        (
            num_periods,
            edu_spec,
            optim_paras,
            num_draws_emax,
            is_debug,
            is_interpolated,
            num_points_interp,
            is_myopic,
            num_agents_sim,
            num_draws_prob,
            tau,
            seed_sim,
            num_agents_est,
            optimizer_options,
            file_sim,
            num_types,
            num_paras,
        ) = dist_class_attributes(
            respy_obj,
            "num_periods",
            "edu_spec",
            "optim_paras",
            "num_draws_emax",
            "is_debug",
            "is_interpolated",
            "num_points_interp",
            "is_myopic",
            "num_agents_sim",
            "num_draws_prob",
            "tau",
            "seed_sim",
            "num_agents_est",
            "optimizer_options",
            "file_sim",
            "num_types",
            "num_paras",
        )

        data_array = process_dataset(respy_obj).to_numpy()
        min_idx = edu_spec["max"] + 1
        shocks_cholesky = optim_paras["shocks_cholesky"]
        coeffs_common = optim_paras["coeffs_common"]
        coeffs_home = optim_paras["coeffs_home"]
        coeffs_edu = optim_paras["coeffs_edu"]
        coeffs_a = optim_paras["coeffs_a"]
        coeffs_b = optim_paras["coeffs_b"]
        delta = optim_paras["delta"]

        type_spec_shares = optim_paras["type_shares"]
        type_spec_shifts = optim_paras["type_shifts"]

        # Write out random components and interpolation grid to align the three
        # implementations.
        max_draws = max(num_agents_sim, num_draws_emax, num_draws_prob)
        write_types(type_spec_shares, num_agents_sim)
        write_edu_start(edu_spec, num_agents_sim)
        write_draws(num_periods, max_draws)
        write_lagged_start(num_agents_sim)

        periods_draws_emax = read_draws(num_periods, num_draws_emax)
        periods_draws_prob = read_draws(num_periods, num_draws_prob)
        periods_draws_sims = read_draws(num_periods, num_agents_sim)

        fort, _ = resfort_interface(respy_obj, "simulate")

        state_space = pyth_solve(
            is_interpolated,
            num_points_interp,
            num_periods,
            is_myopic,
            is_debug,
            periods_draws_emax,
            edu_spec,
            optim_paras,
            file_sim,
            num_types,
        )

        (
            states_all,
            mapping_state_idx,
            periods_rewards_systematic,
            periods_emax,
        ) = state_space._get_fortran_counterparts()

        py = (
            periods_rewards_systematic,
            state_space.states_per_period,
            mapping_state_idx,
            periods_emax,
            states_all,
        )

        f2py = fort_debug.wrapper_solve(
            is_interpolated,
            num_points_interp,
            num_draws_emax,
            num_periods,
            is_myopic,
            is_debug,
            periods_draws_emax,
            min_idx,
            edu_spec["start"],
            edu_spec["max"],
            coeffs_common,
            coeffs_a,
            coeffs_b,
            coeffs_edu,
            coeffs_home,
            shocks_cholesky,
            delta,
            file_sim,
            max_states_period,
            num_types,
            type_spec_shares,
            type_spec_shifts,
        )

        for alt in [fort, f2py]:
            for i in range(5):
                assert_allclose(py[i], alt[i])

        (
            states_all,
            mapping_state_idx,
            periods_rewards_systematic,
            periods_emax,
        ) = state_space._get_fortran_counterparts()

        simulated_data = pyth_simulate(
            state_space,
            num_agents_sim,
            periods_draws_sims,
            seed_sim,
            file_sim,
            edu_spec,
            optim_paras,
            is_debug,
        )
        py = simulated_data.copy().fillna(MISSING_FLOAT).values

        # Is is very important to cut the data array down to the size of the estimation
        # sample for the calculation of contributions.
        data_array = py[: num_agents_est * num_periods, :]

        f2py = fort_debug.wrapper_simulate(
            periods_rewards_systematic,
            mapping_state_idx,
            periods_emax,
            states_all,
            num_periods,
            num_agents_sim,
            periods_draws_sims,
            seed_sim,
            file_sim,
            edu_spec["start"],
            edu_spec["max"],
            edu_spec["share"],
            edu_spec["lagged"],
            optim_paras["coeffs_common"],
            optim_paras["coeffs_a"],
            optim_paras["coeffs_b"],
            shocks_cholesky,
            delta,
            num_types,
            type_spec_shares,
            type_spec_shifts,
            is_debug,
        )
        assert_allclose(py, f2py)

        # We have to cut the simulated data to `num_agents_est` as the Python
        # implementation calculates the likelihood contributions for all agents in the
        # data.
        simulated_data = simulated_data.loc[
            simulated_data.Identifier.lt(num_agents_est)
        ]

        py = pyth_contributions(
            state_space, simulated_data, periods_draws_prob, tau, optim_paras
        )

        num_obs_agent = np.bincount(simulated_data.Identifier.to_numpy())

        f2py = fort_debug.wrapper_contributions(
            periods_rewards_systematic,
            mapping_state_idx,
            periods_emax,
            states_all,
            data_array,
            periods_draws_prob,
            tau,
            num_periods,
            num_draws_prob,
            num_agents_est,
            num_obs_agent,
            num_types,
            edu_spec["start"],
            edu_spec["max"],
            shocks_cholesky,
            delta,
            type_spec_shares,
            type_spec_shifts,
        )

        assert_allclose(py, f2py)

        # Evaluation of criterion function
        x0 = get_optim_paras(optim_paras, num_paras, "all", is_debug)

        py = pyth_criterion(
            x0,
            is_interpolated,
            num_points_interp,
            is_myopic,
            is_debug,
            simulated_data,
            tau,
            periods_draws_emax,
            periods_draws_prob,
            state_space,
            edu_spec,
        )

        f2py = fort_debug.wrapper_criterion(
            x0,
            is_interpolated,
            num_draws_emax,
            num_periods,
            num_points_interp,
            is_myopic,
            is_debug,
            data_array,
            num_draws_prob,
            tau,
            periods_draws_emax,
            periods_draws_prob,
            states_all,
            state_space.states_per_period,
            mapping_state_idx,
            max_states_period,
            num_agents_est,
            num_obs_agent,
            num_types,
            edu_spec["start"],
            edu_spec["max"],
            edu_spec["share"],
            type_spec_shares,
            type_spec_shifts,
            num_paras,
        )

        assert_allclose(py, f2py)

    def test_6(self):
        """ Further tests for the interpolation routines.
        """
        params_spec, options_spec = generate_random_model()
        respy_obj = RespyCls(params_spec, options_spec)
        respy_obj = simulate_observed(respy_obj)

        # Extract class attributes
        (
            periods_rewards_systematic,
            mapping_state_idx,
            seed_prob,
            periods_emax,
            num_periods,
            states_all,
            num_points_interp,
            edu_spec,
            num_draws_emax,
            is_myopic,
            is_debug,
            is_interpolated,
            optim_paras,
            optimizer_options,
            file_sim,
            num_types,
        ) = dist_class_attributes(
            respy_obj,
            "periods_rewards_systematic",
            "mapping_state_idx",
            "seed_prob",
            "periods_emax",
            "num_periods",
            "states_all",
            "num_points_interp",
            "edu_spec",
            "num_draws_emax",
            "is_myopic",
            "is_debug",
            "is_interpolated",
            "optim_paras",
            "optimizer_options",
            "file_sim",
            "num_types",
        )

        shocks_cholesky = optim_paras["shocks_cholesky"]
        shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)
        coeffs_common = optim_paras["coeffs_common"]
        coeffs_a = optim_paras["coeffs_a"]
        coeffs_b = optim_paras["coeffs_b"]
        delta = optim_paras["delta"]

        # Add some additional objects required for the interfaces to the functions.
        period = np.random.choice(num_periods)

        periods_draws_emax = create_draws(
            num_periods, num_draws_emax, seed_prob, is_debug
        )

        draws_emax_standard = periods_draws_emax[period, :, :]

        draws_emax_risk = transform_disturbances(
            draws_emax_standard, np.zeros(4), shocks_cholesky
        )

        # Initialize Python version and solve.
        state_space = StateSpace(
            num_periods,
            num_types,
            edu_spec["start"],
            edu_spec["max"],
            optim_paras,
        )

        # Integrate periods_emax in state_space
        state_space.emaxs = np.column_stack(
            (
                np.zeros((state_space.num_states, 4)),
                periods_emax[periods_emax != MISSING_FLOAT],
            )
        )

        # Fill emaxs_a - emaxs_home in the requested period
        states_period = state_space.get_attribute_from_period("states", period)

        # Do not get the emaxs from the previous period if we are in the last one.
        if period != state_space.num_periods - 1:
            state_space.emaxs = get_emaxs_of_subsequent_period(
                states_period,
                state_space.indexer,
                state_space.emaxs,
                edu_spec["max"],
            )

        num_states = state_space.states_per_period[period]

        shifts = np.random.randn(4)

        # Slight modification of request which assures that the interpolation code is
        # working.
        num_points_interp = min(num_points_interp, num_states)

        # Get the IS_SIMULATED indicator for the subset of points which are used for the
        # predication model.
        is_simulated = get_simulated_indicator(
            num_points_interp, num_states, period, is_debug
        )

        # Unpack necessary attributes
        rewards_period = state_space.get_attribute_from_period(
            "rewards", period
        )
        emaxs_period = state_space.get_attribute_from_period("emaxs", period)[
            :, :4
        ]

        # Construct the exogenous variables for all points of the state space.
        exogenous, max_emax = get_exogenous_variables(
            rewards_period, emaxs_period, shifts, optim_paras["delta"]
        )

        # Align output between Python and Fortran version.
        exogenous_9 = np.column_stack(
            (exogenous, np.sqrt(exogenous), np.ones(exogenous.shape[0]))
        )
        py = (exogenous_9, max_emax)

        f90 = fort_debug.wrapper_get_exogenous_variables(
            period,
            num_periods,
            num_states,
            periods_rewards_systematic,
            shifts,
            mapping_state_idx,
            periods_emax,
            states_all,
            edu_spec["start"],
            edu_spec["max"],
            delta,
            coeffs_common,
            coeffs_a,
            coeffs_b,
            num_types,
        )

        assert_almost_equal(py[0], f90[0], decimal=15)
        assert_almost_equal(py[1], f90[1], decimal=15)

        # Construct endogenous variable so that the prediction model can be fitted.
        endogenous = get_endogenous_variable(
            rewards_period,
            emaxs_period,
            max_emax,
            is_simulated,
            draws_emax_risk,
            optim_paras["delta"],
        )

        f90 = fort_debug.wrapper_get_endogenous_variable(
            period,
            num_periods,
            num_states,
            periods_rewards_systematic,
            mapping_state_idx,
            periods_emax,
            states_all,
            is_simulated,
            num_draws_emax,
            max_emax,
            draws_emax_risk,
            edu_spec["start"],
            edu_spec["max"],
            shocks_cov,
            delta,
            coeffs_common,
            coeffs_a,
            coeffs_b,
        )
        assert_almost_equal(endogenous, replace_missing_values(f90))

        py = get_predictions(
            endogenous, exogenous, max_emax, is_simulated, file_sim, False
        )

        f90 = fort_debug.wrapper_get_predictions(
            endogenous,
            exogenous_9,
            max_emax,
            is_simulated,
            num_points_interp,
            num_states,
            file_sim,
            False,
        )

        # This assertion fails if a column is all zeros.
        if not exogenous.any(axis=0).any():
            assert_array_almost_equal(py, f90)

    def test_7(self):
        """ This is a special test for shared functions related to the interpolation setup.
        """
        # Impose constraints
        point_constr = {"num_periods": np.random.randint(2, 5)}

        params_spec, options_spec = generate_random_model(
            point_constr=point_constr
        )
        respy_obj = RespyCls(params_spec, options_spec)

        # Extract class attributes
        is_debug, num_periods = dist_class_attributes(
            respy_obj, "is_debug", "num_periods"
        )

        # Write out a grid for the interpolation
        max_states_period = write_interpolation_grid(respy_obj)

        # Draw random request for testing
        num_states = np.random.randint(1, max_states_period)
        candidates = list(range(num_states))

        period = np.random.randint(1, num_periods)
        num_points_interp = np.random.randint(1, num_states + 1)

        # Check function for random choice and make sure that there are no duplicates.
        args = (candidates, num_states, num_points_interp)
        f90 = fort_debug.wrapper_random_choice(*args)
        assert_equal(len(set(f90)), len(f90))
        assert_equal(len(f90), num_points_interp)

        # Check the standard cases of the function.
        args = (num_points_interp, num_states, period, is_debug, num_periods)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)

        assert_equal(len(f90), num_states)
        assert_equal(np.all(f90) in [0, 1], True)

        # Test the standardization across PYTHON, F2PY, and FORTRAN implementations.
        # This is possible as we write out an interpolation grid to disk which is used
        # for both functions.
        base_args = (num_points_interp, num_states, period, is_debug)
        args = base_args
        py = get_simulated_indicator(*args)
        args = base_args + (num_periods,)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)
        assert_array_equal(f90, 1 * py)
        os.unlink(".interpolation.respy.test")

        # Special case where number of interpolation points are same as the number of
        # candidates. In that case the returned indicator should be all TRUE.
        args = (num_states, num_states, period, True, num_periods)
        f90 = fort_debug.wrapper_get_simulated_indicator(*args)
        assert_equal(sum(f90), num_states)

    def test_8(self):
        """ We test the construction of the Cholesky decomposition against each other.
        """
        # Draw a random vector of parameters
        x = np.random.uniform(size=54)

        # Construct the Cholesky decompositions
        py = extract_cholesky(x, info=0)
        fort = fort_debug.wrapper_extract_cholesky(x)

        # Compare the results based on the two methods
        np.testing.assert_equal(fort, py)

    def test_9(self):
        """ Functions related to the scaling procedure.
        """
        for _ in range(1000):
            num_free = np.random.randint(1, 100)
            values = np.random.uniform(-1000.0, 1000.0, size=num_free)
            py = get_scales_magnitudes(values)
            f90 = fort_debug.wrapper_get_scales_magnitude(values, num_free)
            assert_almost_equal(py, f90)

    def test_10(self):
        """ Function that calculates the number of observations by individual.
        """
        for _ in range(2):
            params_spec, options_spec = generate_random_model()
            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj = simulate_observed(respy_obj)

            num_agents_est = respy_obj.get_attr("num_agents_est")

            data_array = process_dataset(respy_obj).to_numpy()

            py = np.bincount(data_array[:, 0].astype(int))
            f90 = fort_debug.wrapper_get_num_obs_agent(
                data_array, num_agents_est
            )

            assert_almost_equal(py, f90)

    def test_11(self):
        """ Function that calculates the conditional type probabilites."""
        for _ in range(1000):
            num_types = np.random.randint(1, 10)
            edu_start = np.random.randint(10, 100)
            type_shares = np.random.normal(0, 1, size=num_types * 2)

            args = [type_shares, np.array([edu_start])]

            py = get_conditional_probabilities(*args)
            fort = fort_debug.wrapper_get_conditional_probabilities(
                *args + [num_types]
            )

            assert_almost_equal(np.sum(py), 1.0)
            assert_almost_equal(py, fort)

    def test_12(self):
        """ Testing the functionality introduced to ensure that the simulation is
        independent of the order of initial conditions and types in the initialization
        file.

        """
        num_elements = np.random.randint(1, 11)

        input_array = np.random.normal(size=num_elements)

        # We first check the sorting implementation.
        py = sorted(input_array)
        f90 = fort_debug.wrapper_sorted(input_array, num_elements)
        assert_equal(py, f90)

        params_spec, options_spec = generate_random_model()
        respy_obj = RespyCls(params_spec, options_spec)

        edu_spec, optim_paras, num_types = dist_class_attributes(
            respy_obj, "edu_spec", "optim_paras", "num_types"
        )

        args = (edu_spec["start"], edu_spec["share"], edu_spec["max"])
        f90 = fort_debug.wrapper_sort_edu_spec(*args)
        py = sort_edu_spec(edu_spec)
        for i, label in enumerate(["start", "share", "max"]):
            assert_equal(py[label], f90[i])

        py = sort_type_info(optim_paras, num_types)
        f90 = fort_debug.wrapper_sort_type_info(
            optim_paras["type_shares"], num_types
        )
        for i, label in enumerate(["order", "shares"]):
            assert_equal(py[label], f90[i])
