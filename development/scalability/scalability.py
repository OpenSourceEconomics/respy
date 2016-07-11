#!/usr/bin/env python
""" We perform a simple scalability exercise to ensure the reliability of the
RESPY package.
"""
import glob

from auxiliary import aggregate_information
from auxiliary import send_notification
from auxiliary import compile_package
from auxiliary import process_tasks
from auxiliary import cleanup

# We need to compile the package as part of this script as otherwise the
# parallel implementation is not available.
# TODO: Is this stil true.
# MPI is not defined on the head
# node on acropolis.
# TODO: Merge a lot of files and capabilities
# TODO: Acropolis does not recognize parallel in wscript
cleanup()

compile_package()

''' Details of the scalability exercise can be specified in the code block
below. Note that only deviations from the benchmark initialization files need to
be addressed.
'''
spec_dict = dict()
spec_dict['maxfun'] = 1000
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
    spec_dict['maxfun'] = 20
    spec_dict['num_draws_emax'] = 5
    spec_dict['num_draws_prob'] = 3
    spec_dict['num_agents'] = 100
    spec_dict['scaling'] = [False, 0.00001]
    spec_dict['num_periods'] = 3

grid_slaves = [0, 2]
for fname in glob.glob('*.ini'):
    process_tasks(spec_dict, fname, grid_slaves)
aggregate_information()
send_notification()
