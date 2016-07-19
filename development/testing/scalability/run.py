#!/usr/bin/env python

import argparse
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_scalability import plot_scalability
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_scalability import run
from auxiliary_shared import cleanup
from config import SPECS


def check_scalability(args, GRID_SLAVES):
    """ Details of the scalability exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    """
    spec_dict = dict()

    if args.is_debug:
        GRID_SLAVES = [0, 2]
        spec_dict['maxfun'] = 200
        spec_dict['num_periods'] = 3
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents_est'] = 100
        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_agents_sim'] = spec_dict['num_agents_est']

    if args.is_finalize:
        plot_scalability()
        return

    cleanup()

    if args.is_compile:
        compile_package()

    for spec in SPECS:
        run(spec_dict, spec + '.ini', GRID_SLAVES)

    aggregate_information('scalability')
    plot_scalability()
    send_notification('scalability')

if __name__ == '__main__':

    # The grid of slaves for the analysis. Note that this needs to be
    # reflected in teh PBS submission script.
    GRID_SLAVES = range(0, 32, 2)

    parser = argparse.ArgumentParser(description='Check scalability of RESPY '
        'package.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='use debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile package')

    parser.add_argument('--finalize', action='store_true', dest='is_finalize',
        default=False, help='just create graph')

    check_scalability(parser.parse_args(), GRID_SLAVES)
