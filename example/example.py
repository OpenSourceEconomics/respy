#!/usr/bin/env python
""" This module runs the tutorial from the online documentation.
"""

import shutil
import glob
import os

import respy

# Initialize an instance of the RespyCls to manage all things related to the
# model specification.
respy_obj = respy.RespyCls('example.ini')

# Simulate the model according to the initial specification.
respy.simulate(respy_obj)

# Estimate the model using the true parameters as the starting values. The
# initialization specifies as single evaluation at the starting values as the
# maxfun flag is set to zero.
x, crit_val = respy.estimate(respy_obj)

# Update the respy class instance, with the new parameters.
respy_obj.update_optim_paras(x)
respy.simulate(respy_obj)

# Store results in directory for later inspection.
os.mkdir('example')
for fname in glob.glob('*.respy.*'):
    shutil.move(fname, 'example')
