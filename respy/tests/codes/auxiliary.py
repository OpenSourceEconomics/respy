""" This module contains auxiliary functions for the PYTEST suite.
"""
import pandas as pd
import numpy as np
import shlex

from respy.python.shared.shared_auxiliary import get_conditional_probabilities
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.solve.solve_auxiliary import StateSpace
from respy.python.shared.shared_constants import DATA_FORMATS_SIM
from respy.python.shared.shared_constants import DATA_LABELS_EST
from respy.python.shared.shared_constants import DATA_LABELS_SIM
from respy.python.simulate.simulate_auxiliary import write_out
from respy.python.shared.shared_constants import OPT_EST_FORT
from respy.python.shared.shared_constants import OPT_EST_PYTH
from respy.python.shared.shared_constants import HUGE_FLOAT

from respy import RespyCls

# module-wide variables
OPTIMIZERS_EST = OPT_EST_FORT + OPT_EST_PYTH


def simulate_observed(respy_obj, is_missings=True):
    """ This function adds two important features of observed datasests: (1) missing
    observations and missing wage information.
    """

    def drop_agents_obs(agent):
        """ We now determine the exact period from which onward the history is truncated and
        cut the simulated dataset down to size.
        """
        start_truncation = np.random.choice(range(1, agent["Period"].max() + 2))
        agent = agent[agent["Period"] < start_truncation]
        return agent

    seed_sim = dist_class_attributes(respy_obj, "seed_sim")

    respy_obj.simulate()

    # It is important to set the seed after the simulation call. Otherwise, the value of the
    # seed differs due to the different implementations of the PYTHON and FORTRAN programs.
    np.random.seed(seed_sim)

    # We read in the baseline simulated dataset.
    data_frame = pd.read_csv(
        "data.respy.dat",
        delim_whitespace=True,
        header=0,
        na_values=".",
        dtype=DATA_FORMATS_SIM,
        names=DATA_LABELS_SIM,
    )

    if is_missings:
        # We truncate the histories of agents. This mimics the frequent empirical fact that we loose
        # track of more and more agents over time.
        data_subset = data_frame.groupby("Identifier").apply(drop_agents_obs)

        # We also want to drop the some wage observations. Note that we might be dealing with a
        # dataset where nobody is working anyway.
        is_working = data_subset["Choice"].isin([1, 2])
        num_drop_wages = int(np.sum(is_working) * np.random.uniform(high=0.5, size=1))
        if num_drop_wages > 0:
            indices = data_subset["Wage"][is_working].index
            index_missing = np.random.choice(indices, num_drop_wages, False)
            data_subset.loc[index_missing, "Wage"] = None
        else:
            pass
    else:
        data_subset = data_frame

    # We can restrict the information to observed entities only.
    data_subset = data_subset[DATA_LABELS_EST]
    write_out(respy_obj, data_subset)

    return respy_obj


def compare_init(fname_base, fname_alt):
    """ This function compares the content of each line of a file without any regards for spaces.
    """
    base_lines = [line.rstrip("\n") for line in open(fname_base, "r")]
    alt_lines = [line.rstrip("\n") for line in open(fname_alt, "r")]

    for i, base_line in enumerate(base_lines):
        if alt_lines[i].replace(" ", "") != base_line.replace(" ", ""):
            return False
    return True


def compare_est_log(base_est_log):
    """ This function is required as the log files can be slightly different for good reasons.
    """
    with open("est.respy.log") as in_file:
        alt_est_log = in_file.readlines()

    for j, _ in enumerate(alt_est_log):
        alt_line, base_line = alt_est_log[j], base_est_log[j]
        list_ = shlex.split(alt_line)

        # We can skip empty lines.
        if not list_:
            continue

        if list_[0] in ["Criterion"]:
            alt_val = float(shlex.split(alt_line)[1])
            base_val = float(shlex.split(base_line)[1])
            np.testing.assert_almost_equal(alt_val, base_val)
        elif list_[0] in ["Time", "Duration", "Identifier"]:
            pass
        else:

            is_floats = False
            try:
                int(shlex.split(alt_line)[0])
                is_floats = True
            except ValueError:
                pass
            # We need to cut the floats some slack. It might very well happen that in the
            # very last digits they are in fact different across the versions.
            if not is_floats:
                assert alt_line == base_line
            else:
                base_floats = get_floats(base_line)
                alt_floats = get_floats(alt_line)
                np.testing.assert_almost_equal(alt_floats, base_floats)


def get_floats(line):
    """ This extracts the floats from the line
    """
    list_ = shlex.split(line)[1:]
    rslt = []
    for val in list_:
        if val == "---":
            val = HUGE_FLOAT
        else:
            val = float(val)
        rslt += [val]
    return rslt


def write_interpolation_grid(file_name):
    """ Write out an interpolation grid that can be used across
    implementations.
    """
    # Process relevant initialization file
    respy_obj = RespyCls(file_name)

    # Distribute class attribute
    num_periods, num_points_interp, edu_spec, num_types = dist_class_attributes(
        respy_obj, "num_periods", "num_points_interp", "edu_spec", "num_types"
    )

    # Determine maximum number of states
    state_space = StateSpace()
    state_space.create_state_space(num_periods, num_types, edu_spec)

    states_number_period = state_space.states_per_period
    max_states_period = max(states_number_period)

    # Initialize container
    booleans = np.tile(True, (max_states_period, num_periods))

    # Iterate over all periods
    for period in range(num_periods):

        # Construct auxiliary objects
        num_states = states_number_period[period]
        any_interpolation = (num_states - num_points_interp) > 0

        # Check applicability
        if not any_interpolation:
            continue

        # Draw points for interpolation
        indicators = np.random.choice(
            range(num_states), size=(num_states - num_points_interp), replace=False
        )

        # Replace indicators
        for i in range(num_states):
            if i in indicators:
                booleans[i, period] = False

    # Write out to file
    np.savetxt(".interpolation.respy.test", booleans, fmt="%s")

    # Some information that is useful elsewhere.
    return max_states_period


def write_draws(num_periods, max_draws):
    """ Write out draws to potentially align the different implementations of
    the model. Note that num draws has to be less or equal to the largest
    number of requested random deviates.
    """
    # Draw standard deviates
    draws_standard = np.random.multivariate_normal(
        np.zeros(4), np.identity(4), (num_periods, max_draws)
    )

    # Write to file to they can be read in by the different implementations.
    with open(".draws.respy.test", "w") as file_:
        for period in range(num_periods):
            for i in range(max_draws):
                fmt = " {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}\n"
                line = fmt.format(*draws_standard[period, i, :])
                file_.write(line)


def write_types(type_shares, num_agents_sim):
    """ We also need to fully control the random types to ensure the comparability between PYTHON
    and FORTRAN simulations.
    """
    # Note that the we simply set the relevant initial condition to a random value. This seems to
    # be sufficient for the testing purposes.
    type_probs = get_conditional_probabilities(
        type_shares, np.random.choice([10, 12, 15])
    )
    types = np.random.choice(len(type_probs), p=type_probs, size=num_agents_sim)
    np.savetxt(".types.respy.test", types, fmt="%i")


def write_edu_start(edu_spec, num_agents_sim):
    """ We also need to fully control the random initial schooling to ensure the comparability
    between PYTHON and FORTRAN simulations.
    """
    types = np.random.choice(
        edu_spec["start"], p=edu_spec["share"], size=num_agents_sim
    )
    np.savetxt(".initial_schooling.respy.test", types, fmt="%i")


def write_lagged_start(num_agents_sim):
    """ We also need to fully control the random initial lagged activity to ensure the
    comparability between PYTHON and FORTRAN simulations.
    """
    types = np.random.choice([3, 4], size=num_agents_sim)
    np.savetxt(".initial_lagged.respy.test", types, fmt="%i")


def get_valid_values(which):
    """ Simply get a valid value.
    """
    assert which in ["amb", "cov", "coeff", "delta"]

    if which in ["amb", "delta"]:
        value = np.random.choice([0.0, np.random.uniform()])
    elif which in ["coeff"]:
        value = np.random.uniform(-0.05, 0.05)
    elif which in ["cov"]:
        value = np.random.uniform(0.05, 1)

    return value


def get_valid_shares(num_groups):
    """ We simply need a valid request for the shares of types summing to one.
    """
    shares = np.random.uniform(size=num_groups)
    shares = shares / np.sum(shares)
    shares = shares.tolist()
    return shares


def transform_to_logit(shares):
    """ This function transform
    """
    denominator = 1.0 / shares[0]
    coeffs = []
    for i in range(len(shares)):
        coeffs += [np.log(shares[i] * denominator)]
        coeffs += [0.0]

    return coeffs
