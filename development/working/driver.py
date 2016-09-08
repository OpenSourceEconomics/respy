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


#sys.path.insert(0, '/home/peisenha/restudToolbox/package/respy/tests/resources')
respy_obj = RespyCls('model.respy.ini')
respy_obj = simulate(respy_obj)
#x, crit_val = estimate(respy_obj)
#print(crit_val)
#assert crit_val == 1.185946601338745
#np.testing.assert_equal(crit_val, 4.1093e-11)
#print(respy_obj.get_attr('periods_emax')[0, 0])


fname = 'test_vault_2.respy.pkl'
tests = pkl.load(open(TEST_RESOURCES_DIR + '/' + fname, 'rb'))


#print(tests)

import json
init_dict, crit_val = tests[0]

print(init_dict.keys())

init_dict['SHOCKS']['coeffs'] = init_dict['SHOCKS']['coeffs'].tolist()
init_dict['SHOCKS']['fixed'] = init_dict['SHOCKS']['fixed'].tolist()
print(init_dict)

with open('regression_vault.respy.out', 'wb') as file_:
    file_.write(str(tests))

with open('regression_vault.respy.out', 'rb') as file_:
    test_ = file_.read()

print(test_[0])