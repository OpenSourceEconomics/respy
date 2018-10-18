from statsmodels.tools.eval_measures import rmse
from copy import deepcopy
import numpy as np
import shlex
import os

from auxiliary_shared import update_class_instance
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import cleanup

from config_analysis import SPEC_DIR

from respy.scripts.scripts_simulate import scripts_simulate
from respy.scripts.scripts_compare import scripts_compare
import respy


def run(spec_dict):
    """ Details of the Monte Carlo exercise can be specified in the code block below. Note that
    only deviations from the benchmark initialization files need to be addressed.
    """

    cleanup()

    os.mkdir("rslt")
    os.chdir("rslt")

    for fname in spec_dict["fnames"]:
        run_single(spec_dict, fname)

    aggregate_information("reliability", spec_dict["fnames"])

    send_notification("reliability")

    os.chdir("../")


def run_single(spec_dict, fname):
    """ Run a version of the Monte Carlo exercise.
    """
    os.mkdir(fname.replace(".ini", ""))
    os.chdir(fname.replace(".ini", ""))

    # We first read in the first specification from the initial paper for our baseline and
    # process only the specified deviations.
    respy_obj = respy.RespyCls(SPEC_DIR + fname)
    update_class_instance(respy_obj, spec_dict)

    respy_obj.write_out()

    # Let us first simulate a baseline sample, store the results for future reference, and start
    # an estimation from the true values.
    x = None

    is_risk = spec_dict["update"]["level"] == 0.00
    num_procs = spec_dict["update"]["num_procs"]

    for request in ["Truth", "Static", "Risk", "Ambiguity"]:

        # If there is no ambiguity in the dataset, then we can skip the AMBIGUITY estimation.
        if is_risk and request == "Ambiguity":
            continue

        # If there is no ambiguity, we will just fit the ambiguity parameter to avoid the
        # computational costs.
        respy_obj.unlock()

        if request == "Truth" and is_risk:
            respy_obj.attr["optim_paras"]["paras_fixed"][1] = True
            # We do only need a subset of the available processors
            respy_obj.attr["num_procs"] = min(num_procs, spec_dict["procs"]["truth"])

        elif request == "Truth" and not is_risk:
            respy_obj.attr["num_procs"] = min(num_procs, spec_dict["procs"]["truth"])

        elif request == "Static":
            # We do only need a subset of the available processors
            respy_obj.attr["num_procs"] = min(num_procs, spec_dict["procs"]["static"])

            # There is no update required, we start with the true parameters from the dynamic
            # ambiguity model.
            respy_obj.attr["optim_paras"]["delta"] = np.array([0.00])
            respy_obj.attr["optim_paras"]["level"] = np.array([0.00])
            respy_obj.attr["optim_paras"]["paras_fixed"][:2] = [True, True]
            respy_obj.attr["optim_paras"]["paras_bounds"][0] = [0.00, None]

        elif request == "Risk":
            # We do only need a subset of the available processors
            num_procs = spec_dict["update"]["num_procs"]
            respy_obj.attr["num_procs"] = min(num_procs, spec_dict["procs"]["risk"])

            # This is an update with the results from the static estimation.
            respy_obj.update_optim_paras(x)

            # Note that we now start with 0.85, which is in the middle of the parameter bounds.
            # Manual testing showed that the program is reliable even if we start at 0.00.
            # However, it does take much more function evaluations.
            respy_obj.attr["optim_paras"]["delta"] = np.array([0.85])
            respy_obj.attr["optim_paras"]["level"] = np.array([0.00])
            respy_obj.attr["optim_paras"]["paras_fixed"][:2] = [False, True]
            respy_obj.attr["optim_paras"]["paras_bounds"][0] = [0.70, 1.00]

        elif request == "Ambiguity":
            # We need the full set of available processors.
            respy_obj.attr["num_procs"] = min(
                num_procs, spec_dict["procs"]["ambiguity"]
            )

            # This is an update with the results from the dynamic risk estimation.
            respy_obj.update_optim_paras(x)

            # We want to be able to start the ambiguity estimation directly from the risk-only
            # case. This requires that we adjust the the starting values for the discount factor
            # manually from zero as it otherwise violates the bounds.
            if respy_obj.attr["optim_paras"]["delta"] == 0.0:
                respy_obj.attr["optim_paras"]["delta"] = np.array([0.85])

            # Note that we start with the maximum level to perturb the system.
            respy_obj.attr["optim_paras"]["level"] = np.array([0.10])
            respy_obj.attr["optim_paras"]["paras_fixed"][:2] = [False, False]
            respy_obj.attr["optim_paras"]["paras_bounds"][0] = [0.70, 1.00]
            respy_obj.attr["optim_paras"]["paras_bounds"][1] = [0.01, 0.15]

        else:
            raise AssertionError

        respy_obj.lock()

        os.mkdir(request.lower())
        os.chdir(request.lower())

        # This ensures that the experience effect is taken care of properly.
        open(".restud.respy.scratch", "w").close()

        respy_obj.write_out()

        simulate_specification(respy_obj, "start", False)
        x, _ = respy_obj.fit()
        simulate_specification(respy_obj, "stop", True, x)

        maxfun = respy_obj.get_attr("maxfun")
        rmse_start, rmse_stop = get_rmse()
        num_evals, num_steps = get_est_log_info()

        os.chdir("../")

        args = (request, rmse_start, rmse_stop, num_evals, num_steps, maxfun)
        record_results(*args)

    os.chdir("../")


def get_est_log_info():
    """ Get the choice probabilities.
    """
    with open("est.respy.info") as in_file:

        for line in in_file.readlines():

            # Split line
            list_ = shlex.split(line)

            # Skip empty lines
            if len(list_) < 4:
                continue

            if list_[2] == "Steps":
                num_steps = int(list_[3])

            if list_[2] == "Evaluations":
                num_evals = int(list_[3])

    # Finishing
    return num_evals, num_steps


def get_choice_probabilities(fname, is_flatten=True):
    """ Get the choice probabilities.
    """

    # Initialize container.
    stats = np.tile(np.nan, (0, 4))

    with open(fname) as in_file:

        for line in in_file.readlines():

            # Split line
            list_ = shlex.split(line)

            # Skip empty lines
            if not list_:
                continue

            # If OUTCOMES is reached, then we are done for good.
            if list_[0] == "Outcomes":
                break

            # Any lines that do not have an integer as their first element
            # are not of interest.
            try:
                int(list_[0])
            except ValueError:
                continue

            # All lines that make it down here are relevant.
            stats = np.vstack((stats, [float(x) for x in list_[1:]]))

    # Return all statistics as a flattened array.
    if is_flatten:
        stats = stats.flatten()

    # Finishing
    return stats


def record_results(label, rmse_start, rmse_stop, num_evals, num_steps, maxfun):
    with open("reliability.respy.info", "a") as out_file:
        # Setting up
        if label == "Truth":
            out_file.write("\n RMSE\n\n")
            fmt = "     {:<10} {:>15} {:>15} {:>15} {:>15}\n\n"
            str_ = fmt.format(*["Setup", "Start", "Stop", "Evals", "Steps"])
            out_file.write(str_)
        fmt = "     {:<10} {:15.10f} {:15.10f} {:15} {:15}\n"
        str_ = fmt.format(*[label, rmse_start, rmse_stop, num_evals, num_steps])
        out_file.write(str_)

        # Add information on maximum allowed evaluations
        if label == "Ambiguity":
            fmt = "\n     {:<7} {:<15}    {:15}\n"
            out_file.write(fmt.format(*["Maximum", "Evaluations", maxfun]))


def get_rmse():
    """ Compute the RMSE based on the relevant parameterization.
    """
    fname = "../truth/start/data.respy.info"
    probs_true = get_choice_probabilities(fname, is_flatten=True)

    fname = "start/data.respy.info"
    probs_start = get_choice_probabilities(fname, is_flatten=True)

    fname = "stop/data.respy.info"
    probs_stop = get_choice_probabilities(fname, is_flatten=True)

    rmse_stop = rmse(probs_stop, probs_true)
    rmse_start = rmse(probs_start, probs_true)

    return rmse_start, rmse_stop


def simulate_specification(respy_obj, subdir, update, paras=None):
    """ Simulate results to assess the estimation performance. Note that we do not update the
    object that is passed in.
    """
    os.mkdir(subdir)
    os.chdir(subdir)

    # This ensures that the experience effect is taken care of properly.
    open(".restud.respy.scratch", "w").close()

    respy_copy = deepcopy(respy_obj)
    if update:
        assert paras is not None
        respy_copy.update_optim_paras(paras)

    # The initialization file is specified to run the actual estimation in the directory above.
    respy_copy.attr["file_est"] = "../" + respy_copy.attr["file_est"]
    respy_copy.write_out()

    # When the observed dataset does not exist, we need to simulate it.
    if not os.path.exists(respy_copy.attr["file_est"]):
        scripts_simulate("model.respy.ini", "data")

    scripts_compare("model.respy.ini", False)

    os.chdir("../")
