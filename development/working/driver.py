#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build '
                     '--debug --without_f2py --without_parallelism') == 0
    os.chdir(cwd)




import shutil

import time


from respy.python.shared.shared_auxiliary import print_init_dict

import numpy as np
from respy.python.solve.solve_ambiguity import criterion_ambiguity, \
    get_worst_case, construct_emax_ambiguity


from respy import RespyCls
from respy import simulate
from respy import estimate

from codes.auxiliary import simulate_observed
from codes.auxiliary import write_draws

from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import dist_class_attributes

#write_draws(5, 5000)
np.random.seed(123)
respy_obj = RespyCls('model.respy.ini')
respy_obj = simulate_observed(respy_obj)
#respy_obj.write_out('test.respy.ini')
#respy_obj = RespyCls('test.respy.ini')
_, crit = estimate(respy_obj)
print crit
np.testing.assert_almost_equal(crit, 0.665863818512904)
