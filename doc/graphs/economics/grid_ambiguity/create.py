#!/usr/bin/env python
""" This script creates a grid of samples with varying levels of ambiguity.
"""

import numpy as np

from auxiliary_economics import process_command_line
from auxiliary_grid import run

''' Execution of module as script.
'''

if __name__ == '__main__':

    is_debug = \
        process_command_line('Create grid with varying levels of ambiguity.')

    # The following key value pairs describe the quantification exercise itself.
    # The risk-only case and the actually estimated value is always included
    # in the exercise.
    spec_dict = dict()
    spec_dict['levels'] = np.linspace(0.0, 0.1, 101).tolist()

    # Modification to the baseline initialization file.
    spec_dict['update'] = dict()
    spec_dict['update']['is_store'] = True

    # The following key value pair describes the debugging setup.
    if is_debug:
        spec_dict['levels'] = [0.0, 0.05, 0.1]
        spec_dict['update']['num_procs'] = 1
        spec_dict['update']['num_periods'] = 3

    run(spec_dict)
