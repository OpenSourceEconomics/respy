import numpy as np

import linecache
import shlex
import os

from respy.python.shared.shared_constants import INADMISSIBILITY_PENALTY
from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_constants import PRINT_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import TINY_FLOAT
from respy.custom_exceptions import MaxfunError
from respy.custom_exceptions import UserError


def get_log_likl(contribs):
    """Aggregate contributions to the likelihood value."""
    if sum(np.abs(contribs) > HUGE_FLOAT) > 0:
        record_warning(5)

    crit_val = -np.mean(np.clip(np.log(contribs), -HUGE_FLOAT, HUGE_FLOAT))

    return crit_val


def distribute_parameters(paras_vec, is_debug=False, info=None, paras_type="optim"):
    """Parse the parameter vector into a dictionary of model quantities.

    Args:
        paras_vec (np.ndarray): 1d numpy array with the parameters
        is_debug (bool): If true, the parameters are checked for validity
        info: ????
        paras_type (str): one of ['econ', 'optim']. A paras_vec of type 'econ' contains the
            the standard deviations and covariances of the shock distribution. This is how
            parameters are represented in the .ini file and the output of .fit().
            A paras_vec of type 'optim' contains the elements of the cholesky factors of the
            covariance matrix of the shock distribution. This type is used internally during
            the likelihood estimation. The default value is 'optim' in order to make the function
            more aligned with Fortran, where we never have to parse 'econ' parameters.

    """
    paras_vec = paras_vec.copy()
    assert paras_type in ["econ", "optim"], "paras_type must be econ or optim."

    if is_debug and paras_type == "optim":
        _check_optimization_parameters(paras_vec)

    pinfo = paras_parsing_information(len(paras_vec))
    paras_dict = {}

    # basic extraction
    for quantity in pinfo:
        start = pinfo[quantity]["start"]
        stop = pinfo[quantity]["stop"]
        paras_dict[quantity] = paras_vec[start:stop]

    # modify the shock_coeffs
    if paras_type == "econ":
        shocks_cholesky = coeffs_to_cholesky(paras_dict["shocks_coeffs"])
    else:
        shocks_cholesky, info = extract_cholesky(paras_vec, info)
    paras_dict["shocks_cholesky"] = shocks_cholesky
    del paras_dict["shocks_coeffs"]

    # overwrite the type information
    type_shares, type_shifts = extract_type_information(paras_vec)
    paras_dict["type_shares"] = type_shares
    paras_dict["type_shifts"] = type_shifts

    # checks
    if is_debug:
        assert check_model_parameters(paras_dict)

    return paras_dict


def get_optim_paras(paras_dict, num_paras, which, is_debug):
    """Stack optimization parameters from a dictionary into a vector of type 'optim'.

    Args:
        paras_dict (dict): dictionary with quantities from which the parameters can be extracted
        num_paras (int): number of parameters in the model (not only free parameters)
        which (str): one of ['free', 'all'], determines whether the resulting parameter vetcor
            contains only free parameters or all parameters.
        is_debug (bool): If True, inputs and outputs are checked for consistency.


    """
    if is_debug:
        assert which in ["free", "all"], 'which must be in ["free", "all"]'
        assert check_model_parameters(paras_dict)

    pinfo = paras_parsing_information(num_paras)
    x = np.tile(np.nan, num_paras)

    start, stop = pinfo["delta"]["start"], pinfo["delta"]["stop"]
    x[start:stop] = paras_dict["delta"]

    start, stop = pinfo["coeffs_common"]["start"], pinfo["coeffs_common"]["stop"]
    x[start:stop] = paras_dict["coeffs_common"]

    start, stop = pinfo["coeffs_a"]["start"], pinfo["coeffs_a"]["stop"]
    x[start:stop] = paras_dict["coeffs_a"]

    start, stop = pinfo["coeffs_b"]["start"], pinfo["coeffs_b"]["stop"]
    x[start:stop] = paras_dict["coeffs_b"]

    start, stop = pinfo["coeffs_edu"]["start"], pinfo["coeffs_edu"]["stop"]
    x[start:stop] = paras_dict["coeffs_edu"]

    start, stop = pinfo["coeffs_home"]["start"], pinfo["coeffs_home"]["stop"]
    x[start:stop] = paras_dict["coeffs_home"]

    start, stop = pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"]
    x[start:stop] = paras_dict["shocks_cholesky"][np.tril_indices(4)]

    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    x[start:stop] = paras_dict["type_shares"][2:]

    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    x[start:stop] = paras_dict["type_shifts"].flatten()[4:]

    if is_debug:
        _check_optimization_parameters(x)

    if which == "free":
        x = [x[i] for i in range(num_paras) if not paras_dict["paras_fixed"][i]]
        x = np.array(x)

    return x


def paras_parsing_information(num_paras):
    """Dictionary with the start and stop indices of each quantity."""
    num_types = int((num_paras - 53) / 6) + 1
    num_shares = (num_types - 1) * 2
    pinfo = {
        "delta": {"start": 0, "stop": 1},
        "coeffs_common": {"start": 1, "stop": 3},
        "coeffs_a": {"start": 3, "stop": 18},
        "coeffs_b": {"start": 18, "stop": 33},
        "coeffs_edu": {"start": 33, "stop": 40},
        "coeffs_home": {"start": 40, "stop": 43},
        "shocks_coeffs": {"start": 43, "stop": 53},
        "type_shares": {"start": 53, "stop": 53 + num_shares},
        "type_shifts": {"start": 53 + num_shares, "stop": num_paras},
    }
    return pinfo


def _check_optimization_parameters(x):
    """Check optimization parameters."""
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    assert np.all(np.isfinite(x))
    return True


def get_conditional_probabilities(type_shares, edu_start):
    """Calculate the conditional choice probabilities.

    The calculation is based on the mulitnomial logit model for one particular
    initial condition.

    """
    # Auxiliary objects
    num_types = int(len(type_shares) / 2)
    probs = np.tile(np.nan, num_types)
    for i in range(num_types):
        lower, upper = i * 2, (i + 1) * 2
        covariate = edu_start > 9
        probs[i] = np.exp(np.sum(type_shares[lower:upper] * [1.0, covariate]))

    # Scaling
    probs = probs / sum(probs)

    return probs


def extract_type_information(x):
    """Extract the information about types from a parameter vector of type 'optim'."""
    pinfo = paras_parsing_information(len(x))

    # Type shares
    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    num_types = int(len(x[start:]) / 6) + 1
    type_shares = x[start:stop]
    type_shares = np.concatenate((np.tile(0.0, 2), type_shares), axis=0)

    # Type shifts
    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    type_shifts = x[start:stop]
    type_shifts = np.reshape(type_shifts, (num_types - 1, 4))
    type_shifts = np.concatenate((np.tile(0.0, (1, 4)), type_shifts), axis=0)

    return type_shares, type_shifts


def extract_cholesky(x, info=None):
    """Extract the cholesky factor of the shock covariance from parameters of type 'optim."""
    pinfo = paras_parsing_information(len(x))
    start, stop = pinfo["shocks_coeffs"]["start"], pinfo["shocks_coeffs"]["stop"]
    shocks_coeffs = x[start:stop]
    dim = number_of_triangular_elements_to_dimensio(len(shocks_coeffs))
    shocks_cholesky = np.zeros((dim, dim))
    shocks_cholesky[np.tril_indices(dim)] = shocks_coeffs

    # Stabilization
    if info is not None:
        info = 0

    # We need to ensure that the diagonal elements are larger than zero during
    # estimation. However, we want to allow for the special case of total
    # absence of randomness for testing with simulated datasets.
    if not (np.count_nonzero(shocks_cholesky) == 0):
        shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
        for i in range(len(shocks_cov)):
            if np.abs(shocks_cov[i, i]) < TINY_FLOAT:
                shocks_cholesky[i, i] = np.sqrt(TINY_FLOAT)
                if info is not None:
                    info = 1

    if info is not None:
        return shocks_cholesky, info
    else:
        return shocks_cholesky, None


def coeffs_to_cholesky(coeffs):
    """Return the cholesky factor of a covariance matrix described by coeffs.

    The function can handle the case of a deterministic model (i.e. where all coeffs = 0)

    Args:
        coeffs (np.ndarray): 1d numpy array that contains the upper triangular elements of a
        covariance matrix whose diagonal elements have been replaced by their square roots.

    """
    dim = dim = number_of_triangular_elements_to_dimensio(len(coeffs))
    shocks = np.zeros((dim, dim))
    shocks[np.triu_indices(dim)] = coeffs
    shocks[np.diag_indices(dim)] **= 2

    shocks_cov = shocks + shocks.T - np.diag(shocks.diagonal())

    if np.count_nonzero(shocks_cov) == 0:
        return np.zeros((dim, dim))
    else:
        return np.linalg.cholesky(shocks_cov)


def cholesky_to_coeffs(shocks_cholesky):
    """ Map the Cholesky factor into the coefficients from the .ini file."""
    shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
    shocks_cov[np.diag_indices(len(shocks_cov))] **= 0.5
    shocks_coeffs = shocks_cov[np.triu_indices(len(shocks_cov))].tolist()
    return shocks_coeffs


def get_total_values(
    period,
    num_periods,
    optim_paras,
    rewards_systematic,
    draws,
    edu_spec,
    mapping_state_idx,
    periods_emax,
    k,
    states_all,
):
    """Get value function of all possible states.

    This is called total value because it is the sum of immediate rewards, including realized
    shocks, and expected future rewards.

    """
    # We need to back out the wages from the total systematic rewards to
    # working in the labor market to add the shock properly.
    exp_a, exp_b, edu, choice_lagged, type_ = states_all[period, k, :]
    wages_systematic = back_out_systematic_wages(
        rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras
    )

    # Initialize containers
    rewards_ex_post = np.tile(np.nan, 4)

    # Calculate ex post rewards
    for j in [0, 1]:
        total_increment = rewards_systematic[j] - wages_systematic[j]
        rewards_ex_post[j] = wages_systematic[j] * draws[j] + total_increment

    for j in [2, 3]:
        rewards_ex_post[j] = rewards_systematic[j] + draws[j]

    # Get future values
    if period != (num_periods - 1):
        emaxs = get_emaxs(
            edu_spec, mapping_state_idx, period, periods_emax, k, states_all
        )
    else:
        emaxs = np.tile(0.0, 4)

    # Calculate total utilities
    total_values = rewards_ex_post + optim_paras["delta"] * emaxs

    # This is required to ensure that the agent does not choose any
    # inadmissible states. If the state is inadmissible emaxs takes value zero.
    if states_all[period, k, 2] >= edu_spec["max"]:
        total_values[2] += INADMISSIBILITY_PENALTY

    # Finishing
    return total_values, rewards_ex_post


def get_emaxs(edu_spec, mapping_state_idx, period, periods_emax, k, states_all):
    """Get emaxs for additional choices."""
    # Distribute state space
    exp_a, exp_b, edu, _, type_ = states_all[period, k, :]

    # Future utilities
    emaxs = np.tile(np.nan, 4)

    # Working in Occupation A
    future_idx = mapping_state_idx[period + 1, exp_a + 1, exp_b, edu, 1 - 1, type_]
    emaxs[0] = periods_emax[period + 1, future_idx]

    # Working in Occupation B
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b + 1, edu, 2 - 1, type_]
    emaxs[1] = periods_emax[period + 1, future_idx]

    # Increasing schooling. Note that adding an additional year of schooling
    # is only possible for those that have strictly less than the maximum level
    # of additional education allowed.
    is_inadmissible = edu >= edu_spec["max"]
    if is_inadmissible:
        emaxs[2] = 0.00
    else:
        future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu + 1, 3 - 1, type_]
        emaxs[2] = periods_emax[period + 1, future_idx]

    # Staying at home
    future_idx = mapping_state_idx[period + 1, exp_a, exp_b, edu, 4 - 1, type_]
    emaxs[3] = periods_emax[period + 1, future_idx]

    # Finishing
    return emaxs


def create_draws(num_periods, num_draws, seed, is_debug):
    """Create the relevant set of draws.

    Handle special case of zero variances as this case is useful for testing.
    The draws are from a standard normal distribution and transformed later in
    the code.
    """
    # Control randomness by setting seed value
    np.random.seed(seed)

    # Draw random deviates from a standard normal distribution or read it from
    # disk. The latter is available to allow for testing across implementation.
    if is_debug and os.path.exists(".draws.respy.test"):
        draws = read_draws(num_periods, num_draws)
    else:
        draws = np.random.multivariate_normal(
            np.zeros(4), np.identity(4), (num_periods, num_draws)
        )

    # Finishing
    return draws


def add_solution(
    respy_obj,
    periods_rewards_systematic,
    states_number_period,
    mapping_state_idx,
    periods_emax,
    states_all,
):
    """Add solution to class instance."""
    respy_obj.unlock()
    respy_obj.set_attr("periods_rewards_systematic", periods_rewards_systematic)
    respy_obj.set_attr("states_number_period", states_number_period)
    respy_obj.set_attr("mapping_state_idx", mapping_state_idx)
    respy_obj.set_attr("periods_emax", periods_emax)
    respy_obj.set_attr("states_all", states_all)
    respy_obj.set_attr("is_solved", True)
    respy_obj.lock()
    return respy_obj


def replace_missing_values(arguments):
    """Replace MISSING_FLOAT with NAN.

    Note that the output argument is of type float in the case missing values
    are found.

    """
    # Antibugging
    assert isinstance(arguments, tuple) or isinstance(arguments, np.ndarray)

    if isinstance(arguments, np.ndarray):
        arguments = (arguments,)

    rslt = tuple()

    for argument in arguments:
        # Transform to float array to evaluate missing values.
        argument_internal = np.asfarray(argument)

        # Determine missing values
        is_missing = argument_internal == MISSING_FLOAT
        if np.any(is_missing):
            # Replace missing values
            argument = np.asfarray(argument)
            argument[is_missing] = np.nan

        rslt += (argument,)

    # Aligning interface.
    if len(rslt) == 1:
        rslt = rslt[0]

    # Finishing
    return rslt


def check_model_parameters(optim_paras):
    """Check the integrity of all model parameters."""
    # Auxiliary objects
    num_types = len(optim_paras["type_shifts"])

    # Checks for all arguments
    keys = [
        "coeffs_a",
        "coeffs_b",
        "coeffs_edu",
        "coeffs_home",
        "shocks_cholesky",
        "delta",
        "type_shares",
        "type_shifts",
        "coeffs_common",
    ]

    for key in keys:
        assert isinstance(optim_paras[key], np.ndarray), key
        assert np.all(np.isfinite(optim_paras[key]))
        assert optim_paras[key].dtype == "float"
        assert np.all(abs(optim_paras[key]) < PRINT_FLOAT)

    # Check for discount rate
    assert optim_paras["delta"] >= 0

    # Checks for common returns
    assert optim_paras["coeffs_common"].size == 2

    # Checks for occupations
    assert optim_paras["coeffs_a"].size == 15
    assert optim_paras["coeffs_b"].size == 15
    assert optim_paras["coeffs_edu"].size == 7
    assert optim_paras["coeffs_home"].size == 3

    # Checks shock matrix
    assert optim_paras["shocks_cholesky"].shape == (4, 4)
    np.allclose(optim_paras["shocks_cholesky"], np.tril(optim_paras["shocks_cholesky"]))

    # Checks for type shares
    assert optim_paras["type_shares"].size == num_types * 2

    # Checks for type shifts
    assert optim_paras["type_shifts"].shape == (num_types, 4)

    # Finishing
    return True


def dist_class_attributes(respy_obj, *args):
    """Distribute class class attributes.

    Args:
        respy_obj: instance of clsRespy
        args: any number of strings that are keys in clsRespy.attr

    Returns:
        list of values from clsRespy.attr or single value from clsRespy.attr

    """
    ret = [respy_obj.get_attr(arg) for arg in args]
    if len(ret) == 1:
        ret = ret[0]

    return ret


def read_draws(num_periods, num_draws):
    """Read the draws from disk.

    This is only used in the development process.
    """
    # Initialize containers
    periods_draws = np.tile(np.nan, (num_periods, num_draws, 4))

    # Read and distribute draws
    draws = np.array(np.genfromtxt(".draws.respy.test"), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws * period
        upper = lower + num_draws
        periods_draws[period, :, :] = draws[lower:upper, :]

    # Finishing
    return periods_draws


def transform_disturbances(draws, shocks_mean, shocks_cholesky):
    """Transform the standard normal deviates to the relevant distribution."""
    draws_transformed = draws.copy()
    draws_transformed = np.dot(shocks_cholesky, draws_transformed.T).T

    for j in range(4):
        draws_transformed[:, j] = draws_transformed[:, j] + shocks_mean[j]

    for j in range(2):
        draws_transformed[:, j] = np.clip(
            np.exp(draws_transformed[:, j]), 0.0, HUGE_FLOAT
        )

    return draws_transformed


def format_opt_parameters(dict_, pos):
    """Format the values depending on whether they are fixed or estimated."""
    # Initialize baseline line
    val = dict_["coeffs"][pos]
    is_fixed = dict_["fixed"][pos]
    bounds = dict_["bounds"][pos]

    line = ["coeff", val, " ", " "]
    if is_fixed:
        line[-2] = "!"

    # Check if any bounds defined
    if any(x is not None for x in bounds):
        line[-1] = "(" + str(bounds[0]) + "," + str(bounds[1]) + ")"

    # Finishing
    return line


def apply_scaling(x, precond_matrix, request):
    """Apply or revert the preconditioning step."""
    if request == "do":
        out = np.dot(precond_matrix, x)
    elif request == "undo":
        out = np.dot(np.linalg.pinv(precond_matrix), x)
    else:
        raise AssertionError

    return out


def get_est_info():
    """Read the parameters from the last step of a previous estimation run."""

    def _process_value(input_, type_):
        try:
            if type_ == "float":
                value = float(input_)
            elif type_ == "int":
                value = int(input_)
        except ValueError:
            value = "---"

        return value

    # We need to make sure that the updating file actually exists.
    if not os.path.exists("est.respy.info"):
        msg = "Parameter update impossible as "
        msg += "file est.respy.info does not exist"
        raise UserError(msg)

    # Initialize container and ensure a fresh start processing the file
    linecache.clearcache()
    rslt = dict()

    # Value of the criterion function
    line = shlex.split(linecache.getline("est.respy.info", 6))
    for key_ in ["start", "step", "current"]:
        rslt["value_" + key_] = _process_value(line.pop(0), "float")

    # Total number of evaluations and steps
    line = shlex.split(linecache.getline("est.respy.info", 49))
    rslt["num_step"] = _process_value(line[3], "int")

    line = shlex.split(linecache.getline("est.respy.info", 51))
    rslt["num_eval"] = _process_value(line[3], "int")

    # Parameter values
    for i, key_ in enumerate(["start", "step", "current"]):
        rslt["paras_" + key_] = []
        for j in range(13, 99):
            line = shlex.split(linecache.getline("est.respy.info", j))
            if not line:
                break
            rslt["paras_" + key_] += [_process_value(line[i + 1], "float")]
        rslt["paras_" + key_] = np.array(rslt["paras_" + key_])

    return rslt


def remove_scratch(fname):
    """Remove scratch files."""
    if os.path.exists(fname):
        os.unlink(fname)


def check_early_termination(maxfun, num_eval):
    """Check for reasons that require early termination of the optimization.

    We want early termination if the number of function evaluations is already
    at maxfun. This is not strictly enforced in some of the SCIPY algorithms.

    The user can also stop the optimization immediate, but gently by putting
    a file called '.stop.respy.scratch' in the working directory.

    """
    if maxfun == num_eval:
        raise MaxfunError

    if os.path.exists(".stop.respy.scratch"):
        raise MaxfunError


def get_num_obs_agent(data_array, num_agents_est):
    """Get a list with the number of observations for each agent."""
    num_obs_agent = np.tile(0, num_agents_est)
    agent_number = data_array[0, 0]
    num_rows = data_array.shape[0]

    q = 0
    for i in range(num_rows):
        # We need to check whether we are faced with a new agent.
        if data_array[i, 0] != agent_number:
            q += 1
            agent_number = data_array[i, 0]

        num_obs_agent[q] += 1

    return num_obs_agent


def back_out_systematic_wages(
    rewards_systematic, exp_a, exp_b, edu, choice_lagged, optim_paras
):
    """Construct the wage component for the labor market rewards."""
    # Construct covariates needed for the general part of labor market rewards.
    covariates = construct_covariates(exp_a, exp_b, edu, choice_lagged, None, None)

    # First we calculate the general component.
    general, wages_systematic = np.tile(np.nan, 2), np.tile(np.nan, 2)

    covars_general = [1.0, covariates["not_exp_a_lagged"], covariates["not_any_exp_a"]]
    general[0] = np.dot(optim_paras["coeffs_a"][12:], covars_general)

    covars_general = [1.0, covariates["not_exp_b_lagged"], covariates["not_any_exp_b"]]
    general[1] = np.dot(optim_paras["coeffs_b"][12:], covars_general)

    # Second we do the same with the common component.
    covars_common = [covariates["hs_graduate"], covariates["co_graduate"]]
    rewards_common = np.dot(optim_paras["coeffs_common"], covars_common)

    for j in [0, 1]:
        wages_systematic[j] = rewards_systematic[j] - general[j] - rewards_common

    return wages_systematic


def construct_covariates(exp_a, exp_b, edu, choice_lagged, type_, period):
    """ Construction of some additional covariates for the reward calculations.
    """
    covariates = dict()

    # These are covariates that are supposed to capture the entry costs.
    covariates["not_exp_a_lagged"] = int((exp_a > 0) and (choice_lagged != 1))
    covariates["not_exp_b_lagged"] = int((exp_b > 0) and (choice_lagged != 2))
    covariates["work_a_lagged"] = int(choice_lagged == 1)
    covariates["work_b_lagged"] = int(choice_lagged == 2)
    covariates["edu_lagged"] = int(choice_lagged == 3)
    covariates["choice_lagged"] = choice_lagged
    covariates["not_any_exp_a"] = int(exp_a == 0)
    covariates["not_any_exp_b"] = int(exp_b == 0)
    covariates["any_exp_a"] = int(exp_a > 0)
    covariates["any_exp_b"] = int(exp_b > 0)
    covariates["period"] = period
    covariates["exp_a"] = exp_a
    covariates["exp_b"] = exp_b
    covariates["type"] = type_
    covariates["edu"] = edu

    if edu is not None:
        covariates["hs_graduate"] = int(edu >= 12)
        covariates["co_graduate"] = int(edu >= 16)

        cond = (not covariates["edu_lagged"]) and (not covariates["hs_graduate"])
        covariates["is_return_not_high_school"] = int(cond)

        cond = (not covariates["edu_lagged"]) and covariates["hs_graduate"]
        covariates["is_return_high_school"] = int(cond)

    if period is not None:
        covariates["is_minor"] = int(period < 2)
        covariates["is_young_adult"] = int(period in [2, 3, 4])
        covariates["is_adult"] = int(period >= 5)

    return covariates


def calculate_rewards_common(covariates, optim_paras):
    """ Calculate the reward component that is common to all alternatives.
    """
    covars_common = [covariates["hs_graduate"], covariates["co_graduate"]]
    rewards_common = np.dot(optim_paras["coeffs_common"], covars_common)

    return rewards_common


def calculate_rewards_general(covariates, optim_paras):
    """ Calculate the non-skill related reward components.
    """
    rewards_general = np.tile(np.nan, 2)
    covars_general = [1.0, covariates["not_exp_a_lagged"], covariates["not_any_exp_a"]]
    rewards_general[0] = np.dot(optim_paras["coeffs_a"][12:], covars_general)

    covars_general = [1.0, covariates["not_exp_b_lagged"], covariates["not_any_exp_b"]]
    rewards_general[1] = np.dot(optim_paras["coeffs_b"][12:], covars_general)

    return rewards_general


def number_of_triangular_elements_to_dimensio(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Args:
        num (int): The number of upper or lower triangular elements in the matrix


    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)
