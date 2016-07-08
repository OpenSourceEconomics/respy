#!/usr/bin/env python
""" Using the first specification from Keane & Wolpin (1994), we perform
a simple Monte Carlo exercise to ensure the reliability of the implementation.
"""
from multiprocessing import Pool
from functools import partial
import glob
import os

from auxiliary import aggregate_information
from auxiliary import send_notification
from auxiliary import run

os.system('git clean -d -f')
if os.path.exists('monte_carlo.respy.info'):
    os.unlink('monte_carlo.respy.info')

''' Details of the Monte Carlo exercise can be specified in the code block
below. Note that only deviations from the benchmark initialization files need to
be addressed.
'''
spec_dict = dict()
spec_dict['maxfun'] = 2000
spec_dict['num_draws_emax'] = 500
spec_dict['num_draws_prob'] = 200
spec_dict['num_agents'] = 1000
spec_dict['scaling'] = [True, 0.00001]

spec_dict['optimizer_used'] = 'FORT-NEWUOA'

spec_dict['optimizer_options'] = dict()
spec_dict['optimizer_options']['FORT-NEWUOA'] = dict()
spec_dict['optimizer_options']['FORT-NEWUOA']['maxfun'] = spec_dict['maxfun']
spec_dict['optimizer_options']['FORT-NEWUOA']['npt'] = 53
spec_dict['optimizer_options']['FORT-NEWUOA']['rhobeg'] = 1.0
spec_dict['optimizer_options']['FORT-NEWUOA']['rhoend'] = spec_dict['optimizer_options']['FORT-NEWUOA']['rhobeg'] * 1e-6

# Set flag to TRUE for debugging purposes
if True:
    spec_dict['maxfun'] = 60
    spec_dict['num_draws_emax'] = 5
    spec_dict['num_draws_prob'] = 3
    spec_dict['num_agents'] = 100
    spec_dict['scaling'] = [False, 0.00001]
    spec_dict['num_periods'] = 3

''' Run the Monte Carlo exercise using multiple processors.
'''
process_tasks = partial(run, spec_dict)
ret = Pool(3).map(process_tasks, glob.glob('*.ini'))
send_notification()
aggregate_information()
