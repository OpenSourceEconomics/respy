#!/usr/bin/env python
from statsmodels.tools.eval_measures import rmse
import numpy as np
import argparse
import shutil
import os

from respy.python.simulate.simulate_auxiliary import construct_transition_matrix
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.simulate.simulate_auxiliary import format_float
from respy.pre_processing.data_processing import process_dataset
from respy.scripts.scripts_update import scripts_update
from respy.custom_exceptions import UserError
from respy import RespyCls


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file
    is_update = args.is_update

    # Check attributes
    if not os.path.exists(init_file):
        raise UserError("Initialization file does not exist")

    if not os.path.exists("est.respy.info") and is_update:
        raise UserError("Information on parameter values from last step unavailable")

    # Finishing
    return init_file, is_update


def _prepare_initial(data_obs, data_sim, num_agents_est, num_agents_sim):
    """ This function prepares the information about the distribution of initial schooling levels
    in both datasets.
    """
    # First we want to construct a fill list of available initial schooling levels
    obs_info = data_obs["Years_Schooling"][:, 0].value_counts().to_dict()
    sim_info = data_sim["Years_Schooling"][:, 0].value_counts().to_dict()
    initial_levels = sorted(list(set(list(obs_info.keys()) + list(sim_info.keys()))))

    infos = []
    for level in initial_levels:
        # We need to account for the possibility that a particular initial level of schooling
        # is only present in one of the datasets.
        info = [level, None, None]

        if level in obs_info.keys():
            info[1] = obs_info[level] / float(num_agents_est)

        if level in sim_info.keys():
            info[2] = sim_info[level] / float(num_agents_sim)

        infos += [info]

    return infos


def _prepare_wages(data_obs, data_sim, which):
    """ Prepare the results from the wages for the print out.
    """
    if which == "Occupation A":
        choice_ind = 1
    else:
        choice_ind = 2

    rslt = dict()
    for label in ["Observed", "Simulated"]:
        rslt[label] = []
        if label == "Observed":
            data = data_obs
        else:
            data = data_sim
        for period in range(data_obs["Period"].nunique()):
            is_occupation = data["Choice"] == choice_ind
            series = data["Wage"].ix[is_occupation][:, period]
            rslt[label] += [list(series.describe().values)]

    return rslt


def _prepare_choices(data_obs, data_sim):
    """ This function prepares the information about the choice probabilities for easy printing.
    """
    rslt_full = dict()
    rslt_shares = dict()

    for label in ["Observed", "Simulated"]:

        rslt_full[label] = []
        rslt_shares[label] = []

        if label == "Observed":
            data = data_obs
        else:
            data = data_sim

        for period in range(data_obs["Period"].nunique()):
            shares = []
            total = data["Choice"].loc[:, period].count()
            for choice in [1, 2, 3, 4]:
                count = np.sum(data["Choice"].loc[:, period] == choice)
                shares += [count / float(total)]
            rslt_full[label] += [[total] + shares]
            rslt_shares[label] += shares

    # We also prepare the overall RMSE.
    rmse_choice = rmse(rslt_shares["Observed"], rslt_shares["Simulated"])

    return rslt_full, rmse_choice


def scripts_compare(base_init, is_update):
    """ Construct some model fit statistics by comparing the observed and simulated dataset.
    """
    # In case of updating, we create a new initialization file that contains the updated
    # parameter values.
    if is_update:
        init_file = "compare.respy.ini"
        shutil.copy(base_init, init_file)
        scripts_update(init_file)
    else:
        init_file = base_init

    # Read in relevant model specification.
    respy_obj = RespyCls(init_file)
    respy_obj.write_out("compare.respy.ini")

    # Distribute some information for further processing.
    num_periods, num_agents_est, num_agents_sim = dist_class_attributes(
        respy_obj, "num_periods", "num_agents_est", "num_agents_sim"
    )

    # The comparison does make sense when the file of the simulated dataset and estimation dataset
    # are the same. Then the estimation dataset is overwritten by the simulated dataset.
    fname_est = respy_obj.attr["file_est"].split(".")[0]
    fname_sim = respy_obj.attr["file_sim"].split(".")[0]
    if fname_est == fname_sim:
        raise UserError(" Simulation would overwrite estimation dataset")
    data_obs = process_dataset(respy_obj)
    data_sim = respy_obj.simulate()[1]

    if num_periods > 1:
        tf = []
        tf += [construct_transition_matrix(data_obs)]
        tf += [construct_transition_matrix(data_sim)]

    # Distribute class attributes
    max_periods = len(data_obs["Period"].unique())

    # Prepare results
    rslt_initial = _prepare_initial(data_obs, data_sim, num_agents_est, num_agents_sim)
    rslt_choice, rmse_choice = _prepare_choices(data_obs, data_sim)
    rslt_a = _prepare_wages(data_obs, data_sim, "Occupation A")
    rslt_b = _prepare_wages(data_obs, data_sim, "Occupation B")

    with open("compare.respy.info", "w") as file_:

        file_.write("\n Comparing the Observed and Simulated Economy\n\n")

        file_.write("   Number of Periods:      " + str(max_periods) + "\n\n")

        file_.write("\n   Initial Schooling Shares \n\n")
        fmt_ = "{:>15}" * 3 + "\n"
        labels = ["Level", "Observed", "Simulated"]
        file_.write(fmt_.format(*labels) + "\n")
        for info in rslt_initial:
            info[1:] = [format_float(x) for x in info[1:]]
            file_.write(fmt_.format(*info))

        # Comparing the choice distributions
        file_.write("\n\n   Choices \n\n")
        fmt_ = "{:>15}" * 7 + "\n"
        labels = ["Data", "Period", "Count", "White", "Blue", "School", "Home"]
        file_.write(fmt_.format(*labels) + "\n")
        for period in range(max_periods):
            for name in ["Observed", "Simulated"]:
                line = [name, period + 1] + rslt_choice[name][period]
                fmt_ = "{:>15}" * 3 + "{:15.2f}" * 4 + "\n"
                file_.write(fmt_.format(*line))
            file_.write("\n")
        line = "   Overall RMSE {:14.5f}\n".format(rmse_choice)
        file_.write(line)

        # Comparing the transition matrices
        if num_periods > 1:
            file_.write("\n\n   Transition Matrix \n\n")
            fmt_ = "{:>15}" * 6 + "\n\n"
            labels = ["Work A", "Work B", "School", "Home"]
            file_.write(fmt_.format(*["", ""] + labels))
            for i in range(4):
                for j, source in enumerate(["Observed", "Simulated"]):
                    fmt_ = "{:>15}{:>15}" + "{:15.4f}" * 4 + "\n"
                    line = [source, labels[i]] + tf[j][i, :].tolist()
                    file_.write(fmt_.format(*line))
                file_.write("\n")

        # Comparing the wages distributions
        file_.write("\n   Outcomes \n\n")
        fmt_ = "{:>15}" * 8 + "\n"

        labels = []
        labels += ["Data", "Period", "Count", "Mean", "Std."]
        labels += ["25%", "50%", "75%"]

        file_.write(fmt_.format(*labels) + "\n")
        for rslt, name in [(rslt_a, "Occupation A"), (rslt_b, "Occupation B")]:
            file_.write("\n    " + name + " \n\n")
            for period in range(max_periods):
                for label in ["Observed", "Simulated"]:
                    counts = int(rslt[label][period][0])
                    line = [label, period + 1, counts]
                    # The occurrence of NAN requires special care.
                    stats = rslt[label][period][1:]
                    stats = [format_float(x) for x in stats]
                    file_.write(fmt_.format(*line + stats))
                file_.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare observed and simulated economy."
    )

    parser.add_argument(
        "--init",
        action="store",
        dest="init_file",
        default="model.respy.ini",
        help="initialization file",
    )

    parser.add_argument(
        "--update",
        action="store_true",
        dest="is_update",
        default=False,
        help="update parameterizations",
    )

    init_file, is_update = dist_input_arguments(parser)

    scripts_compare(init_file, is_update)
