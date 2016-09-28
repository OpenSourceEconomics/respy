#!/usr/bin/env python
from auxiliary_reliability import run

if __name__ == '__main__':

    # The following key value pairs are the requested updates from the
    # baseline initialization file.
    spec_dict = dict()
    spec_dict['file_est'] = '../truth/start/data.respy.dat'
    spec_dict['optimizer_used'] = 'FORT-BOBYQA'
    spec_dict['measure'] = 'kl'
    spec_dict['num_procs'] = 3
    spec_dict['amb_max'] = 0.01
    spec_dict['level'] = 0.004
    spec_dict['maxfun'] = 1000

    # The following key value pair describes the debugging setup.
    spec_dict['scaling'] = [True, 0.00001, 1e-6]
    spec_dict['num_periods'] = 3
    spec_dict['num_procs'] = 1
    spec_dict['maxfun'] = 1

    run(spec_dict)
