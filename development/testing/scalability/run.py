#!/usr/bin/env python
""" We perform a simple scalability exercise to ensure the reliability of the
RESPY package.
"""
import argparse
import glob
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_scalability import run
from auxiliary_shared import cleanup


def check_scalability(args):

    cleanup()

    #compile_package()

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
    if args.is_debug:
        spec_dict['maxfun'] = 20
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents'] = 100
        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_periods'] = 3

    grid_slaves = [0, 2]
    for fname in glob.glob('*.ini'):
        run(spec_dict, fname, grid_slaves)

    aggregate_information('scalability')
    send_notification('scalability')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check reliability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='debug specification')

    check_scalability(parser.parse_args())
