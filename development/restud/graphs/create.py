""" This module produces the graphs for the lecture on the RESTUD economies.
"""


# standard library
import shutil
import sys
import os

# lexical analysis
import shlex

# numerical computing
import numpy as np

# hand-crafted plots
from auxiliary import *

# module-wide variables
HOME = os.environ['ROBUPY'] + '/development/restud/graphs'

# PYTHONPATH
sys.path.insert(0, os.environ['ROBUPY'])
sys.path.insert(0, HOME)

# project library
from robupy import read

# Start with a clean slate
os.chdir(HOME)

# Wage functions
coeffs = {}

for spec in ['One', 'Two', 'Three']:

    os.chdir('../simulation/robupy/data_' + spec.lower())

    coeffs[spec] = dict()

    robupy_obj = read('model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')

    for label in ['A', 'B']:
        coeffs[spec][label] = init_dict[label]

    os.chdir(HOME)

# Read number of states from RESTUD output file
num_states = []
with open('../simulation/dp3asim/data_one/otest.txt', 'r') as output_file:
    for line in output_file.readlines():
        if ('kmax(t)' in line):
            num_states += [int(shlex.split(line)[-1])]

# Create and show plot
plot_dimension_state_space(num_states)

# Determine wages for varying level of experience in each occupation
for spec in ['One', 'Two', 'Three']:
    for which in ['A', 'B']:
        x, y = np.meshgrid(range(40), range(40))
        z = np.tile(np.nan, (40,40))
        for i in range(40):
            for j in range(40):
                z[i,j] = return_to_experience(i, j, coeffs[spec], which)

        # Create and show plot
        plot_return_experience(x, y, z, spec)
