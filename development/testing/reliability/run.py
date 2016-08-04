#!/usr/bin/env python

from multiprocessing import Pool
from functools import partial
import argparse
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_reliability import run
from auxiliary_shared import cleanup
from config import SPECS


def check_reliability(maxfun, is_compile, is_debug):
    """ Details of the Monte Carlo exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    """
    spec_dict = dict()
    spec_dict['num_procs'] = 5
    spec_dict['maxfun'] = maxfun

    if is_debug:
        spec_dict['maxfun'] = 60
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents_est'] = 100
        spec_dict['num_agents_sim'] = spec_dict['num_agents_est']
        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_periods'] = 3
        spec_dict['num_procs'] = 1

    cleanup()

    if is_compile:
        compile_package()

    tasks = []
    for spec in SPECS:
        tasks += [spec + '.ini']

    process_tasks = partial(run, spec_dict)
    Pool(3).map(process_tasks, tasks)

    aggregate_information('reliability')
    send_notification('reliability')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check reliability of RESPY '
        'package.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='use debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile package')

    # Setting the maximum number of evaluations.
    maxfun = 3000

    args = parser.parse_args()

    is_compile, is_debug = args.is_compile, args.is_debug

    check_reliability(maxfun, is_compile, is_debug)
