#!/usr/bin/env python
""" This module produces the graphs about the second RESTUD economy.
"""

# standard library
import numpy as np

import shutil
import sys
import os

np.random.seed(123)
import respy

sys.path.insert(0, '../_modules')
from auxiliary_samples import get_choice_probabilities
from auxiliary_samples import plot_return_experience
from auxiliary_samples import plot_return_education
from auxiliary_samples import return_to_experience
from auxiliary_samples import plot_choice_patterns
from auxiliary_samples import return_to_education

# module-wide variables
HOME = os.path.dirname(os.path.realpath(__file__))
SPECS = ['one', 'two', 'three']
SPEC_DIR = '../../../example/'
INIT_FILE = '../../../graphs.respy.ini'

if __name__ == '__main__':

    # Cleanup results from a previous run and prepare the directory structure.
    if os.path.exists('rslt'):
        shutil.rmtree('rslt')
    os.mkdir('rslt')

    os.chdir('rslt')

    # Determine choice patterns over time by simulating the samples and
    # processing the summary information about the dataset.
    respy_obj = respy.RespyCls(INIT_FILE)
    respy_obj.unlock()
    respy_obj.set_attr('num_procs', 3)
    respy_obj.lock()

    respy.simulate(respy_obj)

    choice_probabilities = get_choice_probabilities('data.respy.info')
    plot_choice_patterns(choice_probabilities)

    # Process the initialization files and obtaining the parameters of the
    # reward function that determine the returns to experience and education.
    respy_obj = respy.RespyCls(INIT_FILE)
    model_paras = respy_obj.get_attr('model_paras')

    coeffs = dict()
    for label in ['a', 'b']:
        coeffs[label] = model_paras['coeffs_' + label]

    # Determine wages for varying level of experience in each occupation.
    z = dict()
    for which in ['a', 'b']:
        x, y = np.meshgrid(range(20), range(20))
        z[which] = np.tile(np.nan, (20, 20))
        for i in range(20):
            for j in range(20):
                args = [i, j, coeffs, which]
                z[which][i, j] = return_to_experience(*args)
    plot_return_experience(x, y, z)

    # Determine wages for varying years of education in each occupation.
    xvals, yvals = range(10, 21), dict()
    for which in ['a', 'b']:
        yvals[which] = []
        for edu in xvals:
            yvals[which] += [return_to_education(edu, coeffs, which)]
    plot_return_education(xvals, yvals)
