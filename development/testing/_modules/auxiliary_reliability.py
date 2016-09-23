from statsmodels.tools.eval_measures import rmse
from copy import deepcopy
import numpy as np
import shlex
import os

from config import SPEC_DIR

from auxiliary_shared import update_class_instance
import respy


def run_single(spec_dict, fname):
    """ Run a version of the Monte Carlo exercise.
    """
    os.mkdir(fname.replace('.ini', ''))
    os.chdir(fname.replace('.ini', ''))

    # We first read in the first specification from the initial paper for our
    # baseline and process only the specified deviations.
    respy_obj = respy.RespyCls(SPEC_DIR + fname)
    update_class_instance(respy_obj, spec_dict)

    # Let us first simulate a baseline sample, store the results for future
    # reference, and start an estimation from the true values.
    for request in ['Truth', 'Static', 'Risk', 'Ambiguity']:

        respy_obj.unlock()

        if request == 'Truth':
            pass

        elif request == 'Static':
            # There is no update required, we start with the true parameters
            # from the dynamic ambiguity model.
            respy_obj.set_attr('delta', 0.00)
            respy_obj.attr['model_paras']['level'] = np.array([0.00])
            respy_obj.attr['paras_fixed'][0] = True

        elif request == 'Risk':
            # This is an update with the results from the static estimation.
            respy_obj.update_model_paras(x)

            respy_obj.set_attr('delta', 0.95)
            respy_obj.attr['model_paras']['level'] = np.array([0.00])
            respy_obj.attr['paras_fixed'][0] = True

        elif request == 'Ambiguity':
            # This is an update with the results from the dynamic risk
            # estimation.
            respy_obj.update_model_paras(x)

            respy_obj.set_attr('delta', 0.95)
            respy_obj.attr['model_paras']['level'] = np.array([0.00])
            respy_obj.attr['paras_fixed'][0] = False
        else:
            raise AssertionError

        respy_obj.lock()

        os.mkdir(request.lower()), os.chdir(request.lower())

        respy_obj.write_out()

        simulate_specification(respy_obj, 'start', False)
        x, _ = respy.estimate(respy_obj)
        simulate_specification(respy_obj, 'stop', True, x)

        maxfun = respy_obj.get_attr('maxfun')
        rmse_start, rmse_stop = get_rmse()
        num_evals, num_steps = get_est_log_info()

        os.chdir('../')

        args = (request, rmse_start, rmse_stop, num_evals, num_steps, maxfun)
        record_results(*args)

    os.chdir('../')


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
            fmt = '     {:<10} {:>15} {:>15} {:>15} {:>15}\n\n'
            str_ = fmt.format(*['Setup', 'Start', 'Stop', 'Evals', 'Steps'])
            out_file.write(str_)
        fmt = '     {:<10} {:15.10f} {:15.10f} {:15} {:15}\n'
        str_ = fmt.format(*[label, rmse_start, rmse_stop, num_evals, num_steps])
        out_file.write(str_)

        # Add information on maximum allowed evaluations
        if label == 'Ambiguity':
            fmt = '\n     {:<7} {:<15}    {:15}\n'
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
