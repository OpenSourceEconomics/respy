#!/usr/bin/env python
""" Using the first specification from Keane & Wolpin (1994), we perform
a simple Monte Carlo exercise to ensure the reliability of the implementation.
"""

# standard library
import os

# project library
import respy

###############################################################################
# SPECIFICATION FOR MONTE-CARLO EXERCISE
###############################################################################
MAXFUN = 10
NUM_DRAWS_EMAX = 50
NUM_DRAWS_PROB = 10

NUM_AGENTS = 100
NUM_PROCS = 3


OPTIMIZER = 'FORT-NEWUOA'
NPT = 28
RHOBEG = 2000
RHOEND = RHOBEG * 1e-6

OPTIMIZER_OPTIONS = dict()
OPTIMIZER_OPTIONS['FORT-NEWUOA'] = dict()
OPTIMIZER_OPTIONS['FORT-NEWUOA']['maxfun'] = MAXFUN
OPTIMIZER_OPTIONS['FORT-NEWUOA']['npt'] = NPT
OPTIMIZER_OPTIONS['FORT-NEWUOA']['rhobeg'] = float(RHOBEG)
OPTIMIZER_OPTIONS['FORT-NEWUOA']['rhoend'] = float(RHOEND)

###############################################################################
###############################################################################

os.system('git clean -d -f')

# We first read in the first specification from the initial paper for our
# baseline.
respy_obj = respy.RespyCls('kw_data_one.ini')

respy_obj.unlock()
respy_obj.set_attr('optimizer_options', OPTIMIZER_OPTIONS)
respy_obj.set_attr('num_draws_emax', NUM_DRAWS_EMAX)
respy_obj.set_attr('num_draws_prob', NUM_DRAWS_PROB)
respy_obj.set_attr('num_agents_est', NUM_AGENTS)
respy_obj.set_attr('num_agents_sim', NUM_AGENTS)
respy_obj.set_attr('optimizer_used', OPTIMIZER)

respy_obj.set_attr('num_procs', NUM_PROCS)
if NUM_PROCS > 1:
    respy_obj.set_attr('is_parallel', True)
else:
    respy_obj.set_attr('is_parallel', False)

respy_obj.lock()

# Let us first simulate a baseline sample and store the results for future
# reference.
os.mkdir('correct'), os.chdir('correct')
respy.simulate(respy_obj)
respy_obj.unlock()
respy_obj.set_attr('maxfun', 0)
respy_obj.lock()
respy_obj.write_out()
respy.estimate(respy_obj)
os.chdir('../')

# Now we will estimate a misspecified model on this dataset assuming that
# agents are myopic.
os.mkdir('static'), os.chdir('static')
respy_obj.unlock()
respy_obj.set_attr('file_est', '../correct/data.respy')
respy_obj.set_attr('delta', 0.00)
respy_obj.set_attr('maxfun', MAXFUN)
respy_obj.lock()

respy_obj.write_out()
x, _ = respy.estimate(respy_obj)
respy_obj.update_model_paras(x)
respy.simulate(respy_obj)
os.chdir('../')

# Using the results from the misspecified model as starting values, we see
# whether we can obtain the initial values.
os.mkdir('dynamic'), os.chdir('dynamic')

respy_obj.unlock()
respy_obj.set_attr('delta', 0.95)
respy_obj.lock()

respy_obj.write_out()
x, _ = respy.estimate(respy_obj)

respy_obj.update_model_paras(x)
respy.simulate(respy_obj)
os.chdir('../')
