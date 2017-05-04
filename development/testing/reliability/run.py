#!/usr/bin/env python
from auxiliary_shared import process_command_line_arguments
from auxiliary_reliability import run

if __name__ == '__main__':

    is_debug = process_command_line_arguments('Run reliability exercise for the package')

    # The following key value pairs describe the quantification exercise itself.
    spec_dict = dict()
    spec_dict['fnames'] = ['kw_data_one_initial.ini']

    # The following key-value pairs are the requested updates from the baseline initialization
    # file.
    spec_dict['update'] = dict()

    spec_dict['update']['is_store'] = True
    spec_dict['update']['file_est'] = '../truth/start/data.respy.dat'
    spec_dict['update']['num_draws_prob'] = 200
    spec_dict['update']['num_draws_emax'] = 500
    spec_dict['update']['num_procs'] = 10
    spec_dict['update']['maxfun'] = 1500
    spec_dict['update']['level'] = 0.00

    # The following key-value pair sets the number of processors for each of the estimations.
    # This is required as the maximum number of useful cores varies drastically depending on the
    # model. The requested number of processors is never larger than the one specified as part of
    # the update dictionary.
    spec_dict['procs'] = dict()
    spec_dict['procs']['ambiguity'] = 4
    spec_dict['procs']['static'] = 3
    spec_dict['procs']['truth'] = 2
    spec_dict['procs']['risk'] = 1

    # The following key-value pair describes the debugging setup.
    if is_debug:
        spec_dict['update']['num_periods'] = 3
        spec_dict['update']['num_procs'] = 4
        spec_dict['update']['level'] = 0.00
        spec_dict['update']['maxfun'] = 0

    run(spec_dict)
