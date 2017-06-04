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
from respy._scripts.scripts_compare import scripts_compare
from codes.auxiliary import simulate_observed
from codes.auxiliary import write_draws

from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.process.process_python import process
#write_draws(5, 5000)
np.random.seed(123)








#print 'running with types'
# This ensures that the experience effect is taken care of properly.
#open('.restud.respy.scratch', 'w').close()

    #respy_obj.write_out('test.respy.ini')
#respy_obj = RespyCls('truth.respy.ini')


num_agents_sim = dist_class_attributes(respy_obj, 'num_agents_sim')

# Simulate a dataset
simulate_observed(respy_obj)

# Iterate over alternative implementations
base_x, base_val = None, None

num_periods = init_dict['BASICS']['periods']
type_shares = init_dict['TYPE_SHARES']['coeffs']

write_draws(num_periods, max_draws)
write_types(type_shares, num_agents_sim)

for version in ['FORTRAN', 'PYTHON']:

    respy_obj.unlock()

    respy_obj.set_attr('version', version)

    respy_obj.lock()

    x, val = estimate(respy_obj)

    # Check for the returned parameters.
    if base_x is None:
        base_x = x
    np.testing.assert_allclose(base_x, x)

    # Check for the value of the criterion function.
    if base_val is None:
        base_val = val
    np.testing.assert_allclose(base_val, val)


#respy_obj = RespyCls('stop.respy.ini')
#_, crit = estimate(respy_obj)
#print crit

#if respy_obj.get_attr('version') == 'PYTHON':
#    np.testing.assert_almost_equal(crit, 4.283019001562922)
#else:
#    np.testing.assert_almost_equal(crit, 3.326039372111592)
