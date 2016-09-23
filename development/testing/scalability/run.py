#!/usr/bin/env python
from auxiliary_scalability import run

if __name__ == '__main__':

    # The following key value pairs describe the quantification exercise itself.
    spec_dict = dict()
    spec_dict['slaves'] = [0, 2, 4]

    # The following key value pairs are the requested updates from the
    # baseline initialization file.
    spec_dict['update'] = dict()
    spec_dict['update']['file_est'] = '../data.respy.dat'
    spec_dict['update']['scaling'] = [False, 0.00001]
    spec_dict['update']['is_debug'] = False
    spec_dict['update']['measure'] = 'kl'
    spec_dict['update']['level'] = 0.05
    spec_dict['update']['maxfun'] = 0

    # The following key value pair describes the debugging setup.
    spec_dict['update']['num_periods'] = 3

    run(spec_dict)
