#!/usr/bin/env python
import argparse

from auxiliary_scalability import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run scalability exercise for the package')

    parser.add_argument('--debug', action='store_true', dest='is_debug', default=False,
                        help='use debugging specification')

    args = parser.parse_args()
    is_debug = args.is_debug

    # The following key value pairs describe the quantification exercise itself.
    spec_dict = dict()
    spec_dict['slaves'] = [0, 2, 4]

    # The following key value pairs are the requested updates from the
    # baseline initialization file.
    spec_dict['update'] = dict()
    spec_dict['update']['file_est'] = '../data.respy.dat'

    spec_dict['precond_spec'] = dict()
    spec_dict['precond_spec']['type'] = 'identity'
    spec_dict['precond_spec']['minimum'] = 0.00001
    spec_dict['precond_spec']['eps'] = 1e-6

    spec_dict['ambi_spec'] = dict()
    spec_dict['ambi_spec']['measure'] = 'kl'
    spec_dict['ambi_spec']['mean'] = True

    spec_dict['update']['is_store'] = False
    spec_dict['update']['is_debug'] = False
    spec_dict['update']['level'] = 0.00
    spec_dict['update']['maxfun'] = 0

    # The following key value pair describes the debugging setup.
    if is_debug:
        spec_dict['update']['num_periods'] = 3

    run(spec_dict)
