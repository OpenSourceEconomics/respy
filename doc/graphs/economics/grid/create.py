#!/usr/bin/env python
""" This script creates a grid of samples with varying levels of ambiguity.
"""

import numpy as np

from auxiliary_grid import run

''' Execution of module as script.
'''

if __name__ == '__main__':

    # The following key value pairs describe the quantification exercise itself.
    # The risk-only case and the actually estimated value is always included
    # in the exercise.
    spec_dict = dict()
    spec_dict['levels'] = [0.0, 0.004, 0.020]
    # spec_dict['levels'] = np.linspace(0.0, 0.1, 101).tolist()[0.0, 0.004, 0.020]

    # Modification to the baseline initialization file.
    spec_dict['update'] = dict()
    spec_dict['update']['is_store'] = True

    # The following key value pair describes the debugging setup.
    # spec_dict['update']['num_procs'] = 1
    #spec_dict['update']['num_periods'] = 3

    run(spec_dict)
