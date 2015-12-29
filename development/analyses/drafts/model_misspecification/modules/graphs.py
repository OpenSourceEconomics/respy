#!/usr/bin/env python
""" This module contains the functions to plot the results from the model
misspecification exercises.
"""

# standard library
import matplotlib.pylab as plt

import shutil
import shlex
import sys
import os

# module-wide variable
ROBUPY_DIR = os.environ['ROBUPY']
SPEC_DIR = ROBUPY_DIR + '/development/analyses/restud/specifications'

# PYTHONPATH
sys.path.insert(0, ROBUPY_DIR + '/development/tests/random')
sys.path.insert(0, ROBUPY_DIR)

# project library
from robupy import read

# Pull in the baseline specification from the RESTUD directory.
shutil.copy(SPEC_DIR + '/data_one.robupy.ini', 'model.robupy.ini')
init_dict = read('model.robupy.ini').get_attr('init_dict')

# Get information about baseline
rslts = dict()
rslts['0.00'] = init_dict['EDUCATION']['int']

# Get all available results.
directories_string, directories_float = [], []

for candidate in next(os.walk('.'))[1]:

    try:
        float(candidate)
    except ValueError:
        continue

    directories_string += [candidate],
    directories_float += [float(candidate)]

# Iterate over all available results directory.
for dir_ in directories_string:
    # Enter directory
    os.chdir(dir_)
    # Process result file
    with open('misspecification.robupy.log', 'r') as file_:
        for line in file_.readlines():
            # Split line
            list_ = shlex.split(line)
            # Skip empty lines
            if not list_:
                continue
            # Check for relevant keyword
            is_result = (list_[0] == 'Result')
            if not is_result:
                continue
            # Process results
            rslts[dir_] = float(list_[1])
    # Finishing
    os.chdir('../')

# Prepare directory structure and the intercepts for the graphs.
os.mkdir('rslts')
intercepts = [rslts['0.00']]
for str_ in sorted(directories_string):
    intercepts += [rslts[str_]]

ax = plt.figure(figsize=(12, 8)).add_subplot(111)

ax.plot(range(len(rslts.keys())), intercepts, '-k', color='red', linewidth=5)

plt.savefig('rslts/model_misspecification.png', bbox_inches='tight',
            format='png')


