#!/usr/bin/env python
""" We perform a simple Monte Carlo exercise to ensure the reliability of the
RESPY package.
"""
from multiprocessing import Pool
from functools import partial
import argparse
import glob
import sys
import os

sys.path.insert(0, '../_modules')
from auxiliary_reliability import run
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_shared import cleanup
from config import SPEC_DIR

def check_reliability(args):

    cleanup()

    if args.is_compile:
        compile_package()

    ''' Details of the Monte Carlo exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    '''
    spec_dict = dict()
    spec_dict['maxfun'] = 2000
    spec_dict['num_draws_emax'] = 500
    spec_dict['num_draws_prob'] = 200
    spec_dict['num_agents'] = 1000
    spec_dict['scaling'] = [True, 0.00001]
    spec_dict['num_procs'] = 5

    spec_dict['optimizer_used'] = 'FORT-NEWUOA'

    spec_dict['optimizer_options'] = dict()
    spec_dict['optimizer_options']['FORT-NEWUOA'] = dict()
    spec_dict['optimizer_options']['FORT-NEWUOA']['maxfun'] = spec_dict['maxfun']
    spec_dict['optimizer_options']['FORT-NEWUOA']['npt'] = 53
    spec_dict['optimizer_options']['FORT-NEWUOA']['rhobeg'] = 1.0
    spec_dict['optimizer_options']['FORT-NEWUOA']['rhoend'] = spec_dict['optimizer_options']['FORT-NEWUOA']['rhobeg'] * 1e-6

    if args.is_debug:
        spec_dict['maxfun'] = 60
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents'] = 100
        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_periods'] = 3
        spec_dict['num_procs'] = 1

    process_tasks = partial(run, spec_dict)

    tasks = []
    for fname in glob.glob(SPEC_DIR + 'kw_data_*.ini'):
        tasks += [fname.replace(SPEC_DIR, '')]

    Pool(3).map(process_tasks, tasks)
    send_notification('reliability')
    aggregate_information('reliability')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check reliability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile RESPY package')

    check_reliability(parser.parse_args())
