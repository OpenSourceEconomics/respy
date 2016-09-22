#!/usr/bin/env python
import argparse
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_reliability import run
from auxiliary_shared import cleanup


def check_reliability(args, maxfun):
    """ Details of the Monte Carlo exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    """
    spec_dict = dict()
    spec_dict['num_procs'] = 3
    spec_dict['maxfun'] = maxfun
    spec_dict['file_est'] = '../truth/start/data.respy.dat'
    spec_dict['measure'] = 'kl'
    spec_dict['level'] = 0.05

    if args.is_debug:
        spec_dict['maxfun'] = 5
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents_est'] = 100
        spec_dict['num_agents_sim'] = spec_dict['num_agents_est']
        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_periods'] = 3
        spec_dict['num_procs'] = 1

    cleanup()

    if args.is_compile:
        compile_package()

    run(spec_dict, 'kw_data_one.ini')

    aggregate_information('reliability')
    send_notification('reliability')

if __name__ == '__main__':

    # This is a key parameter for the whole reliability exercise.
    maxfun = 5

    parser = argparse.ArgumentParser(description='Check reliability of RESPY '
        'package.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='use debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile package')

    check_reliability(parser.parse_args(), maxfun)
