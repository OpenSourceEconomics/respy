#!/usr/bin/env python
""" We perform a simple Monte Carlo exercise to ensure the reliability of the
RESPY package.
"""
from multiprocessing import Pool
from functools import partial
import argparse
import glob
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_reliability import run
from auxiliary_shared import cleanup
from config import SPECS


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
    spec_dict['num_procs'] = 5
    spec_dict['optimizer_used'] = 'FORT-NEWUOA'

    if args.is_debug:
        spec_dict['maxfun'] = 60
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents_est'] = 100
        spec_dict['num_agents_sim'] = spec_dict['num_agents_est']
        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_periods'] = 3
        spec_dict['num_procs'] = 1

    tasks = []
    for spec in SPECS:
        tasks += [spec + '.ini']

    process_tasks = partial(run, spec_dict)
    Pool(3).map(process_tasks, tasks)

    aggregate_information('reliability')
    send_notification('reliability')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check reliability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile RESPY package')

    check_reliability(parser.parse_args())
