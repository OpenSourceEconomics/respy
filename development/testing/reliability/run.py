#!/usr/bin/env python

from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_reliability import run_single
from auxiliary_shared import cleanup


def run(spec_dict):
    """ Details of the Monte Carlo exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    """

    cleanup()

    run_single(spec_dict, 'kw_data_one.ini')

    aggregate_information('reliability')

    send_notification('reliability')

if __name__ == '__main__':

    # The following key value pairs are the requested updates from the
    # baseline initialization file.
    spec_dict = dict()
    spec_dict['file_est'] = '../truth/start/data.respy.dat'
    spec_dict['measure'] = 'kl'
    spec_dict['num_procs'] = 3
    spec_dict['level'] = 0.004
    spec_dict['maxfun'] = 0

    # The following key value pair describes the debugging setup.
    spec_dict['scaling'] = [False, 0.00001]
    spec_dict['num_periods'] = 3

    run(spec_dict)
