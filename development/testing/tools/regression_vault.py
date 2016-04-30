#!/usr/bin/env python
""" This script creates and writes to disk a list of the input for numerous
regression tests. These are included in the test infrastructure.
"""

# standard library
import pickle as pkl
import numpy as np

import sys
version = str(sys.version_info[0])

if version == '2':
    sys.path.insert(0, '../../../respy/tests')
    from codes.random_init import generate_init
else:
    from respy.tests.codes.random_init import generate_init

from respy.python.shared.shared_auxiliary import print_init_dict

from respy.evaluate import evaluate

from respy import RespyCls
from respy import simulate

num_tests = 1000
fname = 'test_vault_' + version + '.respy.pkl'

tests = []
for i in range(num_tests):
    print('\n Creating test ' + str(i))

    init_dict = generate_init(constraints=None)

    respy_obj = RespyCls('test.respy.ini')

    simulate(respy_obj)

    crit_val = evaluate(respy_obj)

    test = (init_dict, crit_val)

    tests += [test]

    pkl.dump(tests, open(fname, 'wb'))

    # This makes sure that the test can actually be reproduced.
    print('  ... ensuring recomputability.')

    test_load = pkl.load(open(fname, 'rb'))

    init_dict, crit_val = test_load[-1]

    print_init_dict(init_dict)

    respy_obj = RespyCls('test.respy.ini')

    simulate(respy_obj)

    np.testing.assert_almost_equal(evaluate(respy_obj), crit_val)
