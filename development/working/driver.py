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


respy_obj = RespyCls('model.respy.ini')

simulate(respy_obj)
estimate(respy_obj)
raise SystemExit('just reading in')

for _ in range(10000000):

    constr = dict()
    constr['is_estimation'] = True
    constr['flag_ambiguity'] = False
    constr['flag_parallelism'] = False

    init_dict = generate_init(constr)

    version = np.random.choice(['PYTHON', 'FORTRAN'])

    print(version)
    init_dict['PROGRAM']['version'] = version

    if version == 'PYTHON':
        init_dict['ESTIMATION']['optimizer'] = 'SCIPY-LBFGSB'
    else:
        init_dict['ESTIMATION']['optimizer'] = 'FORT-BOBYQA'
        init_dict['PROGRAM']['procs'] = np.random.randint(1, 5)

    print_init_dict(init_dict)

    respy_obj = RespyCls('test.respy.ini')
    simulate(respy_obj)
    estimate(respy_obj)
