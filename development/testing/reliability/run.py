#!/usr/bin/env python
import argparse

from auxiliary_reliability import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run reliability exercise for the package')

    parser.add_argument('--debug', action='store_true', dest='is_debug', default=False,
                        help='use debugging specification')

    args = parser.parse_args()
    is_debug = args.is_debug

    # The following key value pairs are the requested updates from the baseline initialization
    # file.
    spec_dict = dict()
    spec_dict['update'] = dict()

    spec_dict['update']['is_store'] = True
    spec_dict['update']['file_est'] = '../truth/start/data.respy.dat'
    spec_dict['update']['num_procs'] = 10
    spec_dict['update']['maxfun'] = 1500
    spec_dict['update']['level'] = 0.00

    # The following key value pair describes the debugging setup.
    if is_debug:
        spec_dict['update']['num_periods'] = 3
        spec_dict['update']['num_procs'] = 4
        spec_dict['update']['level'] = 0.00
        spec_dict['update']['maxfun'] = 0

    run(spec_dict)
