#!/usr/bin/env python
""" Using the first specification from Keane & Wolpin (1994), we perform
a simple Monte Carlo exercise to ensure the reliability of the implementation.
"""

# standard library
from statsmodels.tools.eval_measures import rmse

import numpy as np
import shlex
import os

# project library
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

###############################################################################
# SPECIFICATION FOR MONTE-CARLO EXERCISE
###############################################################################
MAXFUN = 2
NUM_DRAWS_EMAX = 500
NUM_DRAWS_PROB = 200

NUM_AGENTS = 1000
NUM_PROCS = 1


OPTIMIZER = 'FORT-NEWUOA'
NPT = 53
RHOBEG = 1
RHOEND = RHOBEG * 1e-6

OPTIMIZER_OPTIONS = dict()
OPTIMIZER_OPTIONS['FORT-NEWUOA'] = dict()
OPTIMIZER_OPTIONS['FORT-NEWUOA']['maxfun'] = MAXFUN
OPTIMIZER_OPTIONS['FORT-NEWUOA']['npt'] = NPT
OPTIMIZER_OPTIONS['FORT-NEWUOA']['rhobeg'] = float(RHOBEG)
OPTIMIZER_OPTIONS['FORT-NEWUOA']['rhoend'] = float(RHOEND)

SCALING = [True, 0.00001]

###############################################################################
###############################################################################
os.system('git clean -d -f')

# We first read in the first specification from the initial paper for our
# baseline.
respy_obj = respy.RespyCls('kw_data_one.ini')

respy_obj.unlock()
respy_obj.set_attr('file_est', '../correct/start/data.respy')
respy_obj.set_attr('optimizer_options', OPTIMIZER_OPTIONS)
respy_obj.set_attr('num_draws_emax', NUM_DRAWS_EMAX)
respy_obj.set_attr('num_draws_prob', NUM_DRAWS_PROB)
respy_obj.set_attr('num_agents_est', NUM_AGENTS)
respy_obj.set_attr('num_agents_sim', NUM_AGENTS)
respy_obj.set_attr('optimizer_used', OPTIMIZER)
respy_obj.set_attr('scaling', SCALING)
respy_obj.set_attr('maxfun', MAXFUN)


respy_obj.set_attr('num_procs', NUM_PROCS)
if NUM_PROCS > 1:
    respy_obj.set_attr('is_parallel', True)
else:
    respy_obj.set_attr('is_parallel', False)

respy_obj.lock()

# Let us first simulate a baseline sample, store the results for future
# reference, and start an estimation from the true values.
os.mkdir('correct'), os.chdir('correct')
respy_obj.write_out()

os.mkdir('start'), os.chdir('start')
respy_obj.write_out()
respy.simulate(respy_obj)
os.chdir('../')

x, _ = respy.estimate(respy_obj)

os.mkdir('stop'), os.chdir('stop')
respy_obj.update_model_paras(x)
respy_obj.write_out()
respy.simulate(respy_obj)
os.chdir('../')

probs_true = get_choice_probabilities('start/data.respy.info', is_flatten=True)
probs_stop = get_choice_probabilities('stop/data.respy.info', is_flatten=True)
rmse_stop = rmse(probs_stop, probs_true)

os.chdir('../')

record_results('Correct', 0.00, rmse_stop)

# Now we will estimate a misspecified model on this dataset assuming that
# agents are myopic.
respy_obj.unlock()
respy_obj.set_attr('delta', 0.00)
respy_obj.lock()

os.mkdir('static'), os.chdir('static')
respy_obj.write_out()

os.mkdir('start'), os.chdir('start')
respy_obj.write_out()
respy.simulate(respy_obj)
os.chdir('../')

x, _ = respy.estimate(respy_obj)

os.mkdir('stop'), os.chdir('stop')
respy_obj.update_model_paras(x)
respy_obj.write_out()
respy.simulate(respy_obj)
os.chdir('../')


probs_true = get_choice_probabilities('../correct/start/data.respy.info', is_flatten=True)
probs_start = get_choice_probabilities('start/data.respy.info', is_flatten=True)
probs_stop = get_choice_probabilities('stop/data.respy.info', is_flatten=True)
rmse_stop = rmse(probs_stop, probs_true)
rmse_start = rmse(probs_start, probs_true)

os.chdir('../')

record_results('Static', rmse_start, rmse_stop)

# # Using the results from the misspecified model as starting values, we see
# # whether we can obtain the initial values.
respy_obj.unlock()
respy_obj.set_attr('delta', 0.95)
respy_obj.lock()

os.mkdir('dynamic'), os.chdir('dynamic')
respy_obj.write_out()

os.mkdir('start'), os.chdir('start')
respy_obj.write_out()
respy.simulate(respy_obj)
os.chdir('../')

x, _ = respy.estimate(respy_obj)

os.mkdir('stop'), os.chdir('stop')
respy_obj.update_model_paras(x)
respy_obj.write_out()
respy.simulate(respy_obj)
os.chdir('../')

probs_true = get_choice_probabilities('../correct/start/data.respy.info', is_flatten=True)
probs_start = get_choice_probabilities('start/data.respy.info', is_flatten=True)
probs_stop = get_choice_probabilities('stop/data.respy.info', is_flatten=True)
rmse_stop = rmse(probs_stop, probs_true)
rmse_start = rmse(probs_start, probs_true)

os.chdir('../')

record_results('Dynamic', rmse_start, rmse_stop)
