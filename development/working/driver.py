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



import numpy as np


from respy import RespyCls
from respy import simulate
from respy import estimate

from codes.random_init import generate_init


respy_obj = RespyCls('model.respy.ini')
simulate(respy_obj)
#print('going in')
#x, base = estimate(respy_obj)
#print(base)
#np.testing.assert_almost_equal(0.350116964137, base)
