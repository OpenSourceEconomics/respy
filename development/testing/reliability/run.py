#!/usr/bin/env python
from auxiliary_reliability import run

if __name__ == '__main__':

    # The following key value pairs are the requested updates from the
    # baseline initialization file.
    spec_dict = dict()
    spec_dict['update'] = dict()

    spec_dict['update']['is_store'] = True
    spec_dict['update']['file_est'] = '../truth/start/data.respy.dat'
    spec_dict['update']['num_procs'] = 50
    spec_dict['update']['maxfun'] = 100

    # The following key value pair describes the debugging setup.
    is_debug = False
    if is_debug:
        spec_dict['update']['num_periods'] = 3
        spec_dict['update']['num_procs'] = 1
        spec_dict['update']['maxfun'] = 1

    run(spec_dict)
