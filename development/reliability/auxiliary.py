from statsmodels.tools.eval_measures import rmse

import numpy as np

import shlex
import os

import respy


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
            if list_[0] == 'Outcomes':
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


def record_results(label, rmse_start, rmse_stop):

    with open('monte_carlo.respy.info', 'a') as out_file:
        # Setting up
        if label == 'Correct':
            out_file.write('\n RMSE\n\n')
            fmt = '{:>15} {:>15} {:>15}\n\n'
            out_file.write(fmt.format(*['Setup', 'Start', 'Stop']))
        fmt = '{:>15} {:15.10f} {:15.10f}\n'
        out_file.write(fmt.format(*[label, rmse_start, rmse_stop]))


def get_rmse():

    fname = '../correct/start/data.respy.info'
    probs_true = get_choice_probabilities(fname, is_flatten=True)

    fname = 'start/data.respy.info'
    probs_start = get_choice_probabilities(fname, is_flatten=True)

    fname = 'stop/data.respy.info'
    probs_stop = get_choice_probabilities(fname, is_flatten=True)

    rmse_stop = rmse(probs_stop, probs_true)
    rmse_start = rmse(probs_start, probs_true)

    return rmse_start, rmse_stop


def simulate_specification(respy_obj, subdir, update, paras=None):
    """ Simulate results to assess the estimation performance.
    """
    os.mkdir(subdir), os.chdir(subdir)

    if update:
        assert (paras is not None)
        respy_obj.update_model_paras(paras)

    respy_obj.write_out()
    respy.simulate(respy_obj)
    os.chdir('../')