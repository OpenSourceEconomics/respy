#!/usr/bin/env python

import argparse
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_scalability import run
from auxiliary_shared import cleanup


def check_scalability(args, GRID_SLAVES):
    """ Details of the scalability exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    """
    spec_dict = dict()
    spec_dict['maxfun'] = 0
    spec_dict['scaling'] = [False, 0.00001]

    # Introducing ambiguity requires special care, which is addressed in the
    # called function.
    spec_dict['measure'] = 'kl'
    spec_dict['level'] = 0.05

    if args.is_debug:
        GRID_SLAVES = [0, 2, 4]
        spec_dict['num_periods'] = 40
#        spec_dict['num_draws_emax'] = 5
#        spec_dict['num_draws_prob'] = 3
#        spec_dict['num_agents_est'] = 100
#        spec_dict['num_agents_sim'] = spec_dict['num_agents_est']

    cleanup()

    if args.is_compile:
        compile_package()

    run(spec_dict, 'kw_data_one.ini', GRID_SLAVES)

    aggregate_information('scalability')
    send_notification('scalability')

if __name__ == '__main__':

    # The grid of slaves for the analysis. Note that this needs to be
    # reflected in the PBS submission script.
    GRID_SLAVES = range(0, 4, 2)

    parser = argparse.ArgumentParser(description='Check scalability of RESPY '
        'package.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='use debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile package')

    parser.add_argument('--finalize', action='store_true', dest='is_finalize',
        default=False, help='just create graph')

    check_scalability(parser.parse_args(), GRID_SLAVES)
