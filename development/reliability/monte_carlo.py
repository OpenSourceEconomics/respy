#!/usr/bin/env python
""" Using the first specification from Keane & Wolpin (1994), we perform
a simple Monte Carlo exercise to ensure the reliability of the implementation.
"""

# standard library
from auxiliary import simulate_specification
from auxiliary import record_results
from auxiliary import get_rmse

import respy
import os

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

simulate_specification(respy_obj, 'start', False)
x, _ = respy.estimate(respy_obj)
simulate_specification(respy_obj, 'stop', True, x)

rmse_start, rmse_stop = get_rmse()

os.chdir('../')

record_results('Correct', rmse_start, rmse_stop)

# Now we will estimate a misspecified model on this dataset assuming that
# agents are myopic. This will serve as a form of well behaved starting
# values for the real estimation to follow.
respy_obj.unlock()
respy_obj.set_attr('delta', 0.00)
respy_obj.lock()

os.mkdir('static'), os.chdir('static')
respy_obj.write_out()

simulate_specification(respy_obj, 'start', False)
x, _ = respy.estimate(respy_obj)
simulate_specification(respy_obj, 'stop', True, x)

rmse_start, rmse_stop = get_rmse()

os.chdir('../')

record_results('Static', rmse_start, rmse_stop)

# # Using the results from the misspecified model as starting values, we see
# # whether we can obtain the initial values.
respy_obj.unlock()
respy_obj.set_attr('delta', 0.95)
respy_obj.lock()

os.mkdir('dynamic'), os.chdir('dynamic')
respy_obj.write_out()

simulate_specification(respy_obj, 'start', False)
x, _ = respy.estimate(respy_obj)
simulate_specification(respy_obj, 'stop', True, x)

rmse_start, rmse_stop = get_rmse()

os.chdir('../')

record_results('Dynamic', rmse_start, rmse_stop)
