#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('git clean -d -f; ./waf configure build --debug ') == 0
    os.chdir(cwd)




import shutil

import time



from respy.python.evaluate.evaluate_auxiliary import check_input
from respy.python.evaluate.evaluate_auxiliary import check_output

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_auxiliary import create_draws

from respy import simulate, RespyCls, estimate
import numpy as np

import pickle as pkl
# Import package. The late import is required as the compilation needs to
# take place first.
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict

from respy import RespyCls
from respy import simulate
from respy import estimate

from codes.random_init import generate_init

np.random.seed(123)
#sys.path.insert(0, '/home/peisenha/restudToolbox/package/respy/tests/resources')

for _ in range(1000):
    print(_)
    constr = dict()

    constr['is_estimation'] = True

    generate_init(constr)
    respy_obj = RespyCls('test.respy.ini')
    respy_obj = simulate(respy_obj)
    _, base_val = estimate(respy_obj)

    from respy.scripts.scripts_update import scripts_update
    from respy.scripts.scripts_modify import scripts_modify

    scripts_update('test.respy.ini')

    # Just a single function evaluation at the new starting values.
    respy_obj.unlock()
    respy_obj.set_attr('maxfun', 0)
    respy_obj.lock()

    # Let us fix/free some random parameters.
    action = np.random.choice(['fix', 'free'])
    num_draws = np.random.randint(1, 27)
    identifiers = np.random.choice(range(27), num_draws, replace=False)
    scripts_modify(identifiers, action, 'test.respy.ini')

    _, update_val = estimate(respy_obj)

    np.testing.assert_almost_equal(update_val, base_val)
