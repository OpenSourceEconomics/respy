#!/usr/bin/env python
from auxiliary_shared import process_command_line_arguments
from auxiliary_reliability import run

if __name__ == '__main__':

    is_debug = process_command_line_arguments('Run reliability exercise for the package')

    # The following key value pairs describe the quantification exercise itself.
    spec_dict = dict()
    spec_dict['fnames'] = ['kw_data_one_types.ini', 'kw_data_one.ini']

    # The following key value pairs are the requested updates from the baseline initialization
    # file.
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
