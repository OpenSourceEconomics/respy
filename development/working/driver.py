#!/usr/bin/env python
""" I will now try to run some estimations.
"""


# ROOT DIRECTORY
# standard library
import os
import sys

# ROOT DIRECTORY
from respy.python.estimate.estimate_auxiliary import dist_optim_paras

# project library
from respy.python.evaluate.evaluate_python import pyth_evaluate
from respy.fortran.f2py_library import f2py_evaluate

from respy.fortran.fortran import fort_evaluate
from respy.tests.codes.auxiliary import write_draws

from respy.python.evaluate.evaluate_auxiliary import check_input
from respy.python.evaluate.evaluate_auxiliary import check_output

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import dist_model_paras
from respy.python.shared.shared_auxiliary import create_draws

from respy import simulate, solve, evaluate, estimate, RespyCls


if True:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build --debug') == 0
    os.chdir(cwd)


respy_obj = RespyCls('model.respy.ini')
import numpy as np
base = None
for is_parallel in [True, False]:
    print('\n\n')
    respy_obj.attr['is_parallel'] = is_parallel
    crit_val = evaluate.evaluate(respy_obj)
    if base is None:
        base = crit_val

    np.testing.assert_equal(base, crit_val)