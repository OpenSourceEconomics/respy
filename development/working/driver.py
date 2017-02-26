#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('git clean -d -f; ./waf configure build --debug --without_f2py') \
           == 0
    os.chdir(cwd)




import shutil

import time



import numpy as np


from respy import RespyCls
from respy import simulate
from respy import estimate

from codes.random_init import generate_init

from codes.auxiliary import write_draws

np.random.seed(123)
respy_obj = RespyCls('model.respy.ini')

num_periods = respy_obj.get_attr('num_periods')
max_draws = 354
write_draws(num_periods, max_draws)

for version in ['FORTRAN', 'PYTHON']:
    print(version)
    respy_obj.reset()
    respy_obj.unlock()
    respy_obj.attr['version'] = version
    respy_obj.lock()
    simulate(respy_obj)
    _, crit = estimate(respy_obj)
    print(crit)
    #assert crit == 0.25443574243874


#print('going in')
#x, base =
#print(base)
#np.testing.assert_almost_equal(0.350116964137, base)
