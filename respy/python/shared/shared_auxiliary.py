import numpy as np

import linecache
import shlex
import os

from respy.python.shared.shared_constants import MISSING_FLOAT
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_constants import PRINT_FLOAT
from respy.python.shared.shared_constants import HUGE_FLOAT
from respy.python.shared.shared_constants import TINY_FLOAT
from respy.custom_exceptions import MaxfunError
from respy.custom_exceptions import UserError
from respy.python.shared.shared_constants import INADMISSIBILITY_PENALTY
from numba import guvectorize, njit


def get_log_likl(contribs):
    """Aggregate contributions to the likelihood value.

    Parameters
    ----------
    contribs : np.ndarray
        Array with shape (num_agents_est,).

    Returns
    -------
    crit_val : float

    """
    if np.sum(np.abs(contribs) > HUGE_FLOAT) > 0:
        record_warning(5)

    crit_val = -np.mean(np.clip(np.log(contribs), -HUGE_FLOAT, HUGE_FLOAT))

    return crit_val


def distribute_parameters(
    paras_vec, is_debug=False, info=None, paras_type="optim"
):
    """Parse the parameter vector into a dictionary of model quantities.

    Parameters
    ----------
    paras_vec : np.ndarray
        1d numpy array with the parameters
    is_debug : bool
        If true, the parameters are checked for validity
    info : ????
    paras_type : str
        one of ['econ', 'optim']. A paras_vec of type 'econ' contains the the standard
        deviations and covariances of the shock distribution. This is how parameters are
        represented in the .ini file and the output of .fit(). A paras_vec of type
        'optim' contains the elements of the cholesky factors of the covariance matrix
        of the shock distribution. This type is used internally during the likelihood
        estimation. The default value is 'optim' in order to make the function more
        aligned with Fortran, where we never have to parse 'econ' parameters.

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
        paras_dict (dict): dictionary with quantities from which the parameters can be
            extracted.
        num_paras (int): number of parameters in the model (not only free parameters)
        which (str): one of ['free', 'all'], determines whether the resulting parameter
            vector contains only free parameters or all parameters.
        is_debug (bool): If True, inputs and outputs are checked for consistency.


    """
    if is_debug:
        assert which in ["free", "all"], 'which must be in ["free", "all"]'
        assert check_model_parameters(paras_dict)

    pinfo = paras_parsing_information(num_paras)
    x = np.full(num_paras, np.nan)

    start, stop = pinfo["delta"]["start"], pinfo["delta"]["stop"]
    x[start:stop] = paras_dict["delta"]

    start, stop = (
        pinfo["coeffs_common"]["start"],
        pinfo["coeffs_common"]["stop"],
    )
    x[start:stop] = paras_dict["coeffs_common"]

    start, stop = pinfo["coeffs_a"]["start"], pinfo["coeffs_a"]["stop"]
    x[start:stop] = paras_dict["coeffs_a"]

    start, stop = pinfo["coeffs_b"]["start"], pinfo["coeffs_b"]["stop"]
    x[start:stop] = paras_dict["coeffs_b"]

    start, stop = pinfo["coeffs_edu"]["start"], pinfo["coeffs_edu"]["stop"]
    x[start:stop] = paras_dict["coeffs_edu"]

    start, stop = pinfo["coeffs_home"]["start"], pinfo["coeffs_home"]["stop"]
    x[start:stop] = paras_dict["coeffs_home"]

    start, stop = (
        pinfo["shocks_coeffs"]["start"],
        pinfo["shocks_coeffs"]["stop"],
    )
    x[start:stop] = paras_dict["shocks_cholesky"][np.tril_indices(4)]

    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    x[start:stop] = paras_dict["type_shares"][2:]

    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    x[start:stop] = paras_dict["type_shifts"].flatten()[4:]

    if is_debug:
        _check_optimization_parameters(x)

    if which == "free":
        x = [
            x[i] for i in range(num_paras) if not paras_dict["paras_fixed"][i]
        ]
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


def get_conditional_probabilities(type_shares, edu_starts):
    """Calculate the conditional choice probabilities.

    The calculation is based on the mulitnomial logit model for one particular
    initial condition.

    """
    type_shares = type_shares.reshape(-1, 2)
    covariate = edu_starts > 9
    covariates = np.hstack(
        [np.ones((covariate.shape[0], 1)), covariate.reshape(-1, 1)]
    )
    probs = np.exp(covariates.dot(type_shares.T))
    probs /= probs.sum(axis=1, keepdims=True)

    if edu_starts.shape[0] == 1:
        probs = probs.ravel()

    return probs


def extract_type_information(x):
    """Extract the information about types from a parameter vector of type 'optim'."""
    pinfo = paras_parsing_information(len(x))

    # Type shares
    start, stop = pinfo["type_shares"]["start"], pinfo["type_shares"]["stop"]
    num_types = int(len(x[start:]) / 6) + 1
    type_shares = x[start:stop]
    type_shares = np.hstack((np.zeros(2), type_shares))

    # Type shifts
    start, stop = pinfo["type_shifts"]["start"], pinfo["type_shifts"]["stop"]
    type_shifts = x[start:stop]
    type_shifts = np.reshape(type_shifts, (num_types - 1, 4))
    type_shifts = np.vstack((np.zeros(4), type_shifts))

    return type_shares, type_shifts


def extract_cholesky(x, info=None):
    """Extract the cholesky factor of the shock covariance from parameters of type
    'optim."""
    pinfo = paras_parsing_information(len(x))
    start, stop = (
        pinfo["shocks_coeffs"]["start"],
        pinfo["shocks_coeffs"]["stop"],
    )
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

    The function can handle the case of a deterministic model (i.e. where all coeffs =
    0)

    Args:
        coeffs (np.ndarray): 1d numpy array that contains the upper triangular elements
            of a covariance matrix whose diagonal elements have been replaced by their
            square roots.

    """
    dim = number_of_triangular_elements_to_dimensio(coeffs.shape[0])
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
    shocks_cov = shocks_cholesky.dot(shocks_cholesky.T)
    shocks_cov[np.diag_indices(shocks_cov.shape[0])] **= 0.5
    shocks_coeffs = shocks_cov[np.triu_indices(shocks_cov.shape[0])].tolist()

    return shocks_coeffs


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, f4[:, :], f4[:, :]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, f8[:, :], f8[:, :]",
    ],
    "(m), (n), (n), (p, n), () -> (n, p), (n, p)",
    nopython=True,
    target="cpu",
)
def get_continuation_value_and_ex_post_rewards(
    wages, rewards_systematic, emaxs, draws, delta, cont_value, rew_ex_post
):
    """Calculate the continuation value and ex-post rewards.

    This function is a generalized ufunc which is flexible in the number of individuals
    and draws.

    Parameters
    ----------
    wages : np.ndarray
        Array with shape (2,).
    rewards_systematic : np.ndarray
        Array with shape (4,).
    emaxs : np.ndarray
        Array with shape (4,)
    draws : np.ndarray
        Array with shape (num_draws, 4)
    delta : float
        Discount rate.

    Returns
    -------
    cont_value : np.ndarray
        Array with shape (4, num_draws).
    rew_ex_post : np.ndarray
        Array with shape (4, num_draws)

    Examples
    --------
    This example is only valid to benchmark different implementations, but does not
    represent a use case.

    >>> num_states_in_period = 10000
    >>> num_draws = 500

    >>> delta = np.array(0.9)
    >>> wages = np.random.randn(num_states_in_period, 2)
    >>> rewards = np.random.randn(num_states_in_period, 4)
    >>> draws = np.random.randn(num_draws, 4)
    >>> emaxs = np.random.randn(num_states_in_period, 4)

    >>> get_continuation_value(wages, rewards, draws, emaxs, delta).shape
    (10000, 4, 500)

    """
    num_draws = draws.shape[0]
    num_choices = rewards_systematic.shape[0]
    num_wages = wages.shape[0]

    for i in range(num_draws):
        for j in range(num_choices):
            if j < num_wages:
                rew_ex = (
                    wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
                )
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            cont_value[j, i] = rew_ex + delta * emaxs[j]
            rew_ex_post[j, i] = rew_ex


@guvectorize(
    [
        "f4[:], f4[:], f4[:], f4[:, :], f4, f4[:, :]",
        "f8[:], f8[:], f8[:], f8[:, :], f8, f8[:, :]",
    ],
    "(m), (n), (n), (p, n), () -> (n, p)",
    nopython=True,
    target="cpu",
)
def get_continuation_value(
    wages, rewards_systematic, emaxs, draws, delta, cont_value
):
    """Calculate the continuation value.

    This function is a reduced version of
    :func:`get_continutation_value_and_ex_post_rewards` which does not return ex post
    rewards. The reason is that a second return argument doubles runtime whereas it is
    only needed during simulation.

    """
    num_draws, num_choices = draws.shape
    num_wages = wages.shape[0]

    for i in range(num_draws):
        for j in range(num_choices):
            if j < num_wages:
                rew_ex = (
                    wages[j] * draws[i, j] + rewards_systematic[j] - wages[j]
                )
            else:
                rew_ex = rewards_systematic[j] + draws[i, j]

            cont_value[j, i] = rew_ex + delta * emaxs[j]


@njit(nogil=True)
def get_emaxs_of_subsequent_period(states, indexer, emaxs, edu_max):
    """Get the maxmium utility from the subsequent period.

    This function takes a parent node and looks up the utility from each of the four
    choices in the subsequent period.

    Warning
    -------
    This function must be extremely performant as the lookup is done for each state in a
    state space (except for states in the last period) for each evaluation of the
    optimization of parameters.

    Example
    -------
    This example is first and foremost for benchmarking different implementations, not
    for validation.

    >>> num_periods, num_types = 50, 3
    >>> edu_start, edu_max = [10, 15], 20
    >>> state_space = StateSpace(num_periods, num_types, edu_start, edu_max)
    >>> state_space.emaxs = np.r_[
    ...     np.zeros((state_space.states_per_period[:-1].sum(), 5)),
    ...     np.full((state_space.states_per_period[-1], 5), 10)
    ... ]
    >>> for period in reversed(range(state_space.num_periods - 1)):
    ...     states = state_space.get_attribute_from_period("states", period)
    ...     state_space.emaxs = get_emaxs_of_subsequent_period(
    ...         states, state_space.indexer, state_space.emaxs, edu_max
    ...     )
    ...     state_space.emaxs[:, 4] = state_space.emaxs[:, :4].max()
    >>> assert (state_space.emaxs == 10).mean() >= 0.95

    """
    for i in range(states.shape[0]):
        # Unpack parent state and get index.
        period, exp_a, exp_b, edu, choice_lagged, type_ = states[i]
        k_parent = indexer[period, exp_a, exp_b, edu, choice_lagged - 1, type_]

        # Working in Occupation A in period + 1
        k = indexer[period + 1, exp_a + 1, exp_b, edu, 0, type_]
        emaxs[k_parent, 0] = emaxs[k, 4]

        # Working in Occupation B in period +1
        k = indexer[period + 1, exp_a, exp_b + 1, edu, 1, type_]
        emaxs[k_parent, 1] = emaxs[k, 4]

        # Schooling in period + 1. Note that adding an additional year of schooling is
        # only possible for those that have strictly less than the maximum level of
        # additional education allowed. This condition is necessary as there are states
        # which have reached maximum education. Incrementing education by one would
        # target an inadmissible state.
        if edu >= edu_max:
            emaxs[k_parent, 2] = INADMISSIBILITY_PENALTY
        else:
            k = indexer[period + 1, exp_a, exp_b, edu + 1, 2, type_]
            emaxs[k_parent, 2] = emaxs[k, 4]

        # Staying at home in period + 1
        k = indexer[period + 1, exp_a, exp_b, edu, 3, type_]
        emaxs[k_parent, 3] = emaxs[k, 4]

    return emaxs


def create_draws(num_periods, num_draws, seed, is_debug):
    """Create the relevant set of draws.

    Handle special case of zero variances as this case is useful for testing.
    The draws are from a standard normal distribution and transformed later in
    the code.

    Parameters
    ----------
    num_periods : int
    num_draws : int
    seed : int
    is_debug : bool

    Returns
    -------
    draws : np.array
        Draws with shape (num_periods, num_draws)

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

    return draws


def add_solution(
    respy_obj,
    periods_rewards_systematic,
    states_number_period,
    mapping_state_idx,
    periods_emax,
    states_all,
    *args,
):
    """Add solution to class instance."""
    respy_obj.unlock()
    respy_obj.set_attr(
        "periods_rewards_systematic", periods_rewards_systematic
    )
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
    np.allclose(
        optim_paras["shocks_cholesky"], np.tril(optim_paras["shocks_cholesky"])
    )

    # Checks for type shares
    assert optim_paras["type_shares"].size == num_types * 2

    # Checks for type shifts
    assert optim_paras["type_shifts"].shape == (num_types, 4)

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
    periods_draws = np.full((num_periods, num_draws, 4), np.nan)

    # Read and distribute draws
    draws = np.array(np.genfromtxt(".draws.respy.test"), ndmin=2)
    for period in range(num_periods):
        lower = 0 + num_draws * period
        upper = lower + num_draws
        periods_draws[period, :, :] = draws[lower:upper, :]

    return periods_draws


def transform_disturbances(draws, shocks_mean, shocks_cholesky):
    """Transform the standard normal deviates to the relevant distribution."""
    draws_transformed = draws.dot(shocks_cholesky.T)

    draws_transformed += shocks_mean

    draws_transformed[:, :2] = np.clip(
        np.exp(draws_transformed[:, :2]), 0.0, HUGE_FLOAT
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


def create_covariates(states):
    """Create set of covariates for each state.

    Parameters
    ----------
    states : np.ndarray
        Array with shape (num_states, 6) containing period, exp_a, exp_b, edu,
        choice_lagged and type of each state.

    Returns
    -------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates of each state.

    Examples
    --------
    This example is to benchmark alternative implementations, but even this version does
    not benefit from Numba anymore.

    >>> states, _ = pyth_create_state_space(40, 5, [10], 20)
    >>> covariates = create_covariates(states)
    >>> assert covariates.shape == (states.shape[0], 16)

    """
    covariates = np.zeros((states.shape[0], 16), dtype=np.int8)

    # Experience in A or B, but not in the last period.
    covariates[:, 0] = np.where((states[:, 1] > 0) & (states[:, 4] != 1), 1, 0)
    covariates[:, 1] = np.where((states[:, 2] > 0) & (states[:, 4] != 2), 1, 0)

    # Last occupation was A, B, or education.
    covariates[:, 2] = np.where(states[:, 4] == 1, 1, 0)
    covariates[:, 3] = np.where(states[:, 4] == 2, 1, 0)
    covariates[:, 4] = np.where(states[:, 4] == 3, 1, 0)

    # No experience in A or B.
    covariates[:, 5] = np.where(states[:, 1] == 0, 1, 0)
    covariates[:, 6] = np.where(states[:, 2] == 0, 1, 0)

    # Any experience in A or B.
    covariates[:, 7] = np.where(states[:, 1] > 0, 1, 0)
    covariates[:, 8] = np.where(states[:, 2] > 0, 1, 0)

    # High school or college graduate
    covariates[:, 9] = np.where(states[:, 3] >= 12, 1, 0)
    covariates[:, 10] = np.where(states[:, 3] >= 16, 1, 0)

    # Was not in school last period and is/is not high school graduate
    covariates[:, 11] = np.where(
        (covariates[:, 4] == 0) & (covariates[:, 9] == 0), 1, 0
    )
    covariates[:, 12] = np.where(
        (covariates[:, 4] == 0) & (covariates[:, 9] == 1), 1, 0
    )

    # Define age groups minor (period < 2), young adult (2 <= period <= 4) and adult (5
    # <= period).
    covariates[:, 13] = np.where(states[:, 0] < 2, 1, 0)
    covariates[:, 14] = np.where(np.isin(states[:, 0], [2, 3, 4]), 1, 0)
    covariates[:, 15] = np.where(states[:, 0] >= 5, 1, 0)

    return covariates


def calculate_rewards_common(covariates, coeffs_common):
    """Calculate common rewards.

    Covariates 9 and 10 are indicators for high school and college graduates.

    Parameters
    ----------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates.
    coeffs_common : np.ndarray
        Array with shape (2,) containing coefficients for high school and college
        graduates.

    Returns
    -------
    np.ndarray
        Array with shape (num_states, 1) containing common rewards. Reshaping is
        necessary to broadcast the array over rewards with shape (num_states, 4).

    Example
    -------
    >>> state_space = StateSpace(2, 1, [12, 16], 20)
    >>> coeffs_common = np.array([0.05, 0.6])
    >>> calculate_rewards_common(state_space.covariates, coeffs_common).reshape(-1)
    array([0.05, 0.05, 0.65, 0.65, 0.05, 0.05, 0.05, 0.05, 0.65, 0.65, 0.65,
           0.65])

    """
    return covariates[:, 9:11].dot(coeffs_common).reshape(-1, 1)


def calculate_rewards_general(covariates, coeffs_a, coeffs_b):
    """Calculate general rewards.

    Parameters
    ----------
    covariates : np.ndarray
        Array with shape (num_states, 16) containing covariates.
    coeffs_a : np.ndarray
        Array with shape (3,) containing coefficients.
    coeffs_b : np.ndarray
        Array with shape (3,) containing coefficients.

    Returns
    -------
    rewards_general : np.ndarray
        Array with shape (num_states, 2) containing general rewards of occupation.

    Example
    -------
    >>> state_space = StateSpace(2, 1, [12, 16], 20)
    >>> coeffs_a, coeffs_b = np.array([0.05, 0.6, 0.4]), np.array([0.36, 0.7, 1])
    >>> calculate_rewards_general(
    ...     state_space.covariates, coeffs_a, coeffs_b
    ... ).reshape(-1)
    array([0.45, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45,
           1.36, 0.45, 0.36, 0.05, 1.36, 0.45, 1.36, 0.45, 1.36, 0.45, 0.36,
           0.05, 1.36])

    """
    num_states = covariates.shape[0]
    rewards_general = np.full((num_states, 2), np.nan)

    rewards_general[:, 0] = np.column_stack(
        (np.ones(num_states), covariates[:, [0, 5]])
    ).dot(coeffs_a)
    rewards_general[:, 1] = np.column_stack(
        (np.ones(num_states), covariates[:, [1, 6]])
    ).dot(coeffs_b)

    return rewards_general


def get_valid_bounds(which, value):
    """ Simply get a valid set of bounds."""
    assert which in ["cov", "coeff", "delta", "share"]

    # The bounds cannot be too tight as otherwise the BOBYQA might not start
    # properly.
    if which in ["delta"]:
        upper = np.random.choice([None, value + np.random.uniform(low=0.1)])
        bounds = [max(0.0, value - np.random.uniform(low=0.1)), upper]
    elif which in ["coeff"]:
        upper = np.random.choice([None, value + np.random.uniform(low=0.1)])
        lower = np.random.choice([None, value - np.random.uniform(low=0.1)])
        bounds = [lower, upper]
    elif which in ["cov"]:
        bounds = [None, None]
    elif which in ["share"]:
        bounds = [0.0, None]
    return bounds


def number_of_triangular_elements_to_dimensio(num):
    """Calculate the dimension of a square matrix from number of triangular elements.

    Args:
        num (int): The number of upper or lower triangular elements in the matrix

    Example:
        >>> number_of_triangular_elements_to_dimensio(6)
        3
        >>> number_of_triangular_elements_to_dimensio(10)
        4

    """
    return int(np.sqrt(8 * num + 1) / 2 - 0.5)
