import numpy as np
import pandas as pd
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_auxiliary import check_model_parameters
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.shared.shared_constants import PRINT_FLOAT
from respy.custom_exceptions import UserError
from respy.python.shared.shared_auxiliary import replace_missing_values
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import IS_FORTRAN


def check_model_attributes(attr_dict):
    a = attr_dict

    # Number of parameters
    assert isinstance(a["num_paras"], int)
    assert a["num_paras"] >= 53

    # Parallelism
    assert isinstance(a["num_procs"], int)
    assert a["num_procs"] > 0
    if a["num_procs"] > 1:
        assert a["version"] == "FORTRAN"

    assert isinstance(a["num_procs"], int)
    assert a["num_procs"] > 0
    if a["num_procs"] > 1:
        assert a["version"] == "FORTRAN"
        assert IS_PARALLELISM_MPI

    # Version version of package
    assert a["version"] in ["FORTRAN", "PYTHON"]
    if a["version"] == "FORTRAN":
        assert IS_FORTRAN

    assert isinstance(a["num_threads"], int)
    assert a["num_threads"] >= 1
    if a["num_threads"] >= 2:
        assert a["version"] == "FORTRAN"
        assert IS_PARALLELISM_OMP

    # Debug status
    assert a["is_debug"] in [True, False]

    # Forward-looking agents
    assert a["is_myopic"] in [True, False]

    # Seeds
    for seed in [a["seed_emax"], a["seed_sim"], a["seed_prob"]]:
        assert np.isfinite(seed)
        assert isinstance(seed, int)
        assert seed > 0

    # Number of agents
    for num_agents in [a["num_agents_sim"], a["num_agents_est"]]:
        assert np.isfinite(num_agents)
        assert isinstance(num_agents, int)
        assert num_agents > 0

    # Number of periods
    assert np.isfinite(a["num_periods"])
    assert isinstance(a["num_periods"], int)
    assert a["num_periods"] > 0

    # Number of draws for Monte Carlo integration
    assert np.isfinite(a["num_draws_emax"])
    assert isinstance(a["num_draws_emax"], int)
    assert a["num_draws_emax"] >= 0

    # Debugging mode
    assert a["is_debug"] in [True, False]

    # Window for smoothing parameter
    assert isinstance(a["tau"], float)
    assert a["tau"] > 0

    # Interpolation
    assert a["is_interpolated"] in [True, False]
    assert isinstance(a["num_points_interp"], int)
    assert a["num_points_interp"] > 0

    # Simulation of S-ML
    assert isinstance(a["num_draws_prob"], int)
    assert a["num_draws_prob"] > 0

    # Maximum number of iterations
    assert isinstance(a["maxfun"], int)
    assert a["maxfun"] >= 0

    # Optimizers
    assert a["optimizer_used"] in OPT_EST_FORT + OPT_EST_PYTH

    # Scaling
    assert a["precond_spec"]["type"] in ["identity", "gradient", "magnitudes"]
    for key_ in ["minimum", "eps"]:
        assert isinstance(a["precond_spec"][key_], float)
        assert a["precond_spec"][key_] > 0.0

    # Education
    assert isinstance(a["edu_spec"]["max"], int)
    assert a["edu_spec"]["max"] > 0
    assert isinstance(a["edu_spec"]["start"], list)
    assert len(a["edu_spec"]["start"]) == len(set(a["edu_spec"]["start"]))
    assert all(isinstance(item, int) for item in a["edu_spec"]["start"])
    assert all(item > 0 for item in a["edu_spec"]["start"])
    assert all(item <= a["edu_spec"]["max"] for item in a["edu_spec"]["start"])
    assert all(isinstance(item, float) for item in a["edu_spec"]["share"])
    assert all(0 <= item <= 1 for item in a["edu_spec"]["lagged"])
    assert all(0 <= item <= 1 for item in a["edu_spec"]["share"])
    np.testing.assert_almost_equal(np.sum(a["edu_spec"]["share"]), 1.0, decimal=4)

    # Derivatives
    assert a["derivatives"] in ["FORWARD-DIFFERENCES"]

    # Check model parameters
    check_model_parameters(a["optim_paras"])

    # Check that all parameter values are within the bounds.
    x = get_optim_paras(a["optim_paras"], a["num_paras"], "all", True)

    # It is not clear at this point how to impose parameter constraints on
    # the covariance matrix in a flexible manner. So, either all fixed or
    # none. As a special case, we also allow for all off-diagonal elements
    # to be fixed to zero.
    shocks_coeffs = a["optim_paras"]["shocks_cholesky"][np.tril_indices(4)]
    shocks_fixed = a["optim_paras"]["paras_fixed"][43:53]

    all_fixed = all(is_fixed is False for is_fixed in shocks_fixed)
    all_free = all(is_free is True for is_free in shocks_fixed)

    subset_fixed = [shocks_fixed[i] for i in [1, 3, 4, 6, 7, 8]]
    subset_value = [shocks_coeffs[i] for i in [1, 3, 4, 6, 7, 8]]

    off_diagonal_fixed = all(is_free is True for is_free in subset_fixed)
    off_diagonal_value = all(value == 0.0 for value in subset_value)
    off_diagonal = off_diagonal_fixed and off_diagonal_value

    if not (all_free or all_fixed or off_diagonal):
        raise UserError(" Misspecified constraints for covariance matrix")

    # Discount rate and type shares need to be larger than on at all times.
    for label in ["paras_fixed", "paras_bounds"]:
        assert isinstance(a["optim_paras"][label], list)
        assert len(a["optim_paras"][label]) == a["num_paras"]

    for i in range(1):
        assert a["optim_paras"]["paras_bounds"][i][0] >= 0.00

    for i in range(a["num_paras"]):
        lower, upper = a["optim_paras"]["paras_bounds"][i]
        if lower is not None:
            assert isinstance(lower, float)
            assert lower <= x[i]
            assert abs(lower) < PRINT_FLOAT
        if upper is not None:
            assert isinstance(upper, float)
            assert upper >= x[i]
            assert abs(upper) < PRINT_FLOAT
        if (upper is not None) and (lower is not None):
            assert upper >= lower
        # At this point no bounds for the elements of the covariance matrix
        # are allowed.
        if i in range(43, 53):
            assert a["optim_paras"]["paras_bounds"][i] == [None, None]

    _check_optimizer_options(a["optimizer_options"])


def check_model_solution(attr_dict):
    solution_attributes = [
        "periods_rewards_systematic",
        "states_number_period",
        "mapping_state_idx",
        "periods_emax",
        "states_all",
    ]

    for label in solution_attributes:
        if attr_dict["is_solved"]:
            assert attr_dict[label] is not None
        else:
            assert attr_dict[label] is None

    if attr_dict["is_solved"]:
        # Distribute class attributes
        num_initial = len(attr_dict["edu_spec"]["start"])
        edu_start = attr_dict["edu_spec"]["start"]
        edu_start_max = max(edu_start)
        edu_max = attr_dict["edu_spec"]["max"]
        num_periods = attr_dict["num_periods"]
        num_types = attr_dict["num_types"]

        # Distribute results
        periods_rewards_systematic = attr_dict["periods_rewards_systematic"]
        states_number_period = attr_dict["states_number_period"]
        mapping_state_idx = attr_dict["mapping_state_idx"]
        periods_emax = attr_dict["periods_emax"]
        states_all = attr_dict["states_all"]

        # Replace missing value with NAN. This allows to easily select the
        # valid subsets of the containers
        mapping_state_idx = replace_missing_values(mapping_state_idx)
        states_all = replace_missing_values(states_all)
        periods_rewards_systematic = replace_missing_values(periods_rewards_systematic)
        periods_emax = replace_missing_values(periods_emax)

        # No values can be larger than constraint time. The exception in
        # the lagged schooling variable in the first period, which takes
        # value one but has index zero.
        for period in range(num_periods):
            assert np.nanmax(states_all[period, :, :3]) <= (period + edu_start_max)

        # Lagged schooling can only take value zero or one if finite.
        for period in range(num_periods):
            assert np.nanmax(states_all[period, :, 3]) in [1, 2, 3, 4]
            assert np.nanmin(states_all[period, :, :3]) == 0

        # All finite values have to be larger or equal to zero.
        # The loop is required as np.all evaluates to FALSE for this
        # condition (see NUMPY documentation).
        for period in range(num_periods):
            assert np.all(states_all[period, : states_number_period[period]] >= 0)

        # The maximum of education years is never larger than `edu_max'.
        for period in range(num_periods):
            assert np.nanmax(states_all[period, :, :][:, 2], axis=0) <= edu_max

        # Check for duplicate rows in each period. We only have possible
        # duplicates if there are multiple initial conditions.
        for period in range(num_periods):
            nstates = states_number_period[period]
            assert (
                np.sum(pd.DataFrame(states_all[period, :nstates, :]).duplicated()) == 0
            )

        # Checking validity of state space values. All valid values need
        # to be finite.
        for period in range(num_periods):
            assert np.all(
                np.isfinite(states_all[period, : states_number_period[period]])
            )

        # There are no infinite values in final period.
        assert np.all(np.isfinite(states_all[(num_periods - 1), :, :]))

        # Check the number of states in the first time period.
        num_states_start = num_types * num_initial * 2
        assert np.sum(np.isfinite(mapping_state_idx[0, :, :, :, :])) == num_states_start

        # Check that mapping is defined for all possible realizations of
        # the state space by period. Check that mapping is not defined for
        # all inadmissible values.
        is_infinite = np.full(mapping_state_idx.shape, False)
        for period in range(num_periods):
            nstates = states_number_period[period]
            indices = states_all[period, :nstates, :].astype("int")
            for index in indices:
                assert np.isfinite(
                    mapping_state_idx[
                        period, index[0], index[1], index[2], index[3] - 1, index[4]
                    ]
                )
                is_infinite[
                    period, index[0], index[1], index[2], index[3] - 1, index[4]
                ] = True
        assert np.all(np.isfinite(mapping_state_idx[is_infinite == True]))
        assert np.all(np.isfinite(mapping_state_idx[is_infinite == False])) == False

        # Check the calculated systematic rewards (finite for admissible values
        # and infinite rewards otherwise).
        is_infinite = np.full(periods_rewards_systematic.shape, False)
        for period in range(num_periods):
            for k in range(states_number_period[period]):
                assert np.all(np.isfinite(periods_rewards_systematic[period, k, :]))
                is_infinite[period, k, :] = True
            assert np.all(np.isfinite(periods_rewards_systematic[is_infinite == True]))
            if num_periods > 1:
                assert (
                    np.all(
                        np.isfinite(periods_rewards_systematic[is_infinite == False])
                    )
                    == False
                )

        # Check the expected future value (finite for admissible values
        # and infinite rewards otherwise).
        is_infinite = np.full(periods_emax.shape, False)
        for period in range(num_periods):
            for k in range(states_number_period[period]):
                assert np.all(np.isfinite(periods_emax[period, k]))
                is_infinite[period, k] = True
            assert np.all(np.isfinite(periods_emax[is_infinite == True]))
            if num_periods == 1:
                assert len(periods_emax[is_infinite == False]) == 0
            else:
                assert np.all(np.isfinite(periods_emax[is_infinite == False])) == False


def _check_optimizer_options(optimizer_options):
    """Make sure that all optimizer options are valid."""
    # POWELL's algorithms
    for optimizer in ["FORT-NEWUOA", "FORT-BOBYQA"]:
        maxfun = optimizer_options[optimizer]["maxfun"]
        rhobeg = optimizer_options[optimizer]["rhobeg"]
        rhoend = optimizer_options[optimizer]["rhoend"]
        npt = optimizer_options[optimizer]["npt"]

        for var in [maxfun, npt]:
            assert isinstance(var, int)
            assert var > 0
        for var in [rhobeg, rhoend]:
            assert rhobeg > rhoend
            assert isinstance(var, float)
            assert var > 0

    # FORT-BFGS
    maxiter = optimizer_options["FORT-BFGS"]["maxiter"]
    stpmx = optimizer_options["FORT-BFGS"]["stpmx"]
    gtol = optimizer_options["FORT-BFGS"]["gtol"]
    assert isinstance(maxiter, int)
    assert maxiter > 0
    for var in [stpmx, gtol]:
        assert isinstance(var, float)
        assert var > 0

    # SCIPY-BFGS
    maxiter = optimizer_options["SCIPY-BFGS"]["maxiter"]
    gtol = optimizer_options["SCIPY-BFGS"]["gtol"]
    eps = optimizer_options["SCIPY-BFGS"]["eps"]
    assert isinstance(maxiter, int)
    assert maxiter > 0
    for var in [eps, gtol]:
        assert isinstance(var, float)
        assert var > 0

    # SCIPY-LBFGSB
    maxiter = optimizer_options["SCIPY-LBFGSB"]["maxiter"]
    pgtol = optimizer_options["SCIPY-LBFGSB"]["pgtol"]
    factr = optimizer_options["SCIPY-LBFGSB"]["factr"]
    maxls = optimizer_options["SCIPY-LBFGSB"]["maxls"]
    eps = optimizer_options["SCIPY-LBFGSB"]["eps"]
    m = optimizer_options["SCIPY-LBFGSB"]["m"]

    for var in [pgtol, factr, eps]:
        assert isinstance(var, float)
        assert var > 0
    for var in [m, maxiter, maxls]:
        assert isinstance(var, int)
        assert var >= 0

    # SCIPY-POWELL
    maxiter = optimizer_options["SCIPY-POWELL"]["maxiter"]
    maxfun = optimizer_options["SCIPY-POWELL"]["maxfun"]
    xtol = optimizer_options["SCIPY-POWELL"]["xtol"]
    ftol = optimizer_options["SCIPY-POWELL"]["ftol"]
    assert isinstance(maxiter, int)
    assert maxiter > 0
    assert isinstance(maxfun, int)
    assert maxfun > 0
    assert isinstance(xtol, float)
    assert xtol > 0
    assert isinstance(ftol, float)
    assert ftol > 0
