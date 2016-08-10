from statsmodels.tools.eval_measures import rmse
from copy import deepcopy
import numpy as np
import shlex
import os

from config import SPEC_DIR

import respy


def get_est_log_info():
    """ Get the choice probabilities.
    """
    with open('est.respy.info') as in_file:

        for line in in_file.readlines():

            # Split line
            list_ = shlex.split(line)

            # Skip empty lines
            if len(list_) < 4:
                continue

            if list_[2] == 'Steps':
                num_steps = int(list_[3])

            if list_[2] == 'Evaluations':
                num_evals = int(list_[3])

    # Finishing
    return num_evals, num_steps


def run(spec_dict, fname):
    """ Run a version of the Monte Carlo exercise.
    """
    dirname = fname.replace('.ini', '')
    os.mkdir(dirname)
    os.chdir(dirname)

    # We first read in the first specification from the initial paper for our
    # baseline and process the deviations.
    respy_obj = respy.RespyCls(SPEC_DIR + fname)

    respy_obj.unlock()

    respy_obj.set_attr('file_est', '../truth/start/data.respy.dat')

    for key_ in spec_dict.keys():
        respy_obj.set_attr(key_, spec_dict[key_])

    if respy_obj.attr['num_procs'] > 1:
        respy_obj.set_attr('is_parallel', True)
    else:
        respy_obj.set_attr('is_parallel', False)

    respy_obj.lock()

    maxfun = respy_obj.get_attr('maxfun')

    # Let us first simulate a baseline sample, store the results for future
    # reference, and start an estimation from the true values.
    os.mkdir('truth')
    os.chdir('truth')
    respy_obj.write_out()

    simulate_specification(respy_obj, 'start', False)
    x, _ = respy.estimate(respy_obj)
    simulate_specification(respy_obj, 'stop', True, x)

    rmse_start, rmse_stop = get_rmse()
    num_evals, num_steps = get_est_log_info()

    os.chdir('../')

    record_results('Truth', rmse_start, rmse_stop, num_evals, num_steps, maxfun)

    # Now we will estimate a misspecified model on this dataset assuming that
    # agents are myopic. This will serve as a form of well behaved starting
    # values for the real estimation to follow.
    respy_obj.unlock()
    respy_obj.set_attr('delta', 0.00)
    respy_obj.lock()

    os.mkdir('static')
    os.chdir('static')

    respy_obj.write_out()

    simulate_specification(respy_obj, 'start', False)
    x, _ = respy.estimate(respy_obj)
    simulate_specification(respy_obj, 'stop', True, x)

    rmse_start, rmse_stop = get_rmse()
    num_evals, num_steps = get_est_log_info()

    os.chdir('../')

    record_results('Static', rmse_start, rmse_stop, num_evals, num_steps, maxfun)

    # # Using the results from the misspecified model as starting values, we see
    # # whether we can obtain the initial values.
    respy_obj.update_model_paras(x)

    respy_obj.unlock()
    respy_obj.set_attr('delta', 0.95)
    respy_obj.lock()

    os.mkdir('dynamic')
    os.chdir('dynamic')
    respy_obj.write_out()

    simulate_specification(respy_obj, 'start', False)
    x, _ = respy.estimate(respy_obj)
    simulate_specification(respy_obj, 'stop', True, x)

    rmse_start, rmse_stop = get_rmse()
    num_evals, num_steps = get_est_log_info()

    os.chdir('../')

    record_results('Dynamic', rmse_start, rmse_stop, num_evals, num_steps,
                   maxfun)

    os.chdir('../')


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


def record_results(label, rmse_start, rmse_stop, num_evals, num_steps, maxfun):
    with open('reliability.respy.info', 'a') as out_file:
        # Setting up
        if label == 'Truth':
            out_file.write('\n RMSE\n\n')
            fmt = '{:>15} {:>15} {:>15} {:>15} {:>15}\n\n'
            out_file.write(fmt.format(*['Setup', 'Start', 'Stop', 'Evals', 'Steps']))
        fmt = '{:>15} {:15.10f} {:15.10f} {:15} {:15}\n'
        out_file.write(fmt.format(*[label, rmse_start, rmse_stop, num_evals, num_steps]))

        # Add information on maximum allowed evaluations
        if label == 'Dynamic':
            fmt = '\n{:>15} {:<15} {:15}\n'
            out_file.write(fmt.format(*['Maximum', 'Evaluations', maxfun]))


def get_rmse():
    """ Compute the RMSE based on the relevant parameterization.
    """
    fname = '../truth/start/data.respy.info'
    probs_true = get_choice_probabilities(fname, is_flatten=True)

    fname = 'start/data.respy.info'
    probs_start = get_choice_probabilities(fname, is_flatten=True)

    fname = 'stop/data.respy.info'
    probs_stop = get_choice_probabilities(fname, is_flatten=True)

    rmse_stop = rmse(probs_stop, probs_true)
    rmse_start = rmse(probs_start, probs_true)

    return rmse_start, rmse_stop


def simulate_specification(respy_obj, subdir, update, paras=None):
    """ Simulate results to assess the estimation performance. Note that we do
    not update the object that is passed in.
    """
    os.mkdir(subdir)
    os.chdir(subdir)

    respy_copy = deepcopy(respy_obj)
    if update:
        assert (paras is not None)
        respy_copy.update_model_paras(paras)

    respy_copy.write_out()
    respy.simulate(respy_copy)
    os.chdir('../')
