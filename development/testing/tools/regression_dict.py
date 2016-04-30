#!/usr/bin/env python
""" This script creates and writes to disk a list of the input for numerous
regression tests. These are included in the test infrastructure.
"""

# standard library
import pickle as pkl
import numpy as np

import json

from respy.python.shared.shared_auxiliary import print_init_dict
from respy.tests.codes.random_init import generate_init

from respy.evaluate import evaluate

from respy import RespyCls
from respy import simulate

np.random.seed(213)

NUM_TESTS = 10

tests = []
for _ in range(NUM_TESTS):

    init_dict = generate_init(constraints=None)

    respy_obj = RespyCls('test.respy.ini')

    simulate(respy_obj)

    crit_val = evaluate(respy_obj)

    test = (init_dict, crit_val)

    tests += [test]


    pkl.dump(tests, open('test_list.respy.pkl', 'wb'))

tests = None

# Now we make sure that the tests will pass.
tests = pkl.load(open('test_list.respy.pkl', 'rb'))

for test in tests:

    init_dict, crit_val = test

    print_init_dict(init_dict)

    respy_obj = RespyCls('test.respy.ini')

    simulate(respy_obj)

    np.testing.assert_almost_equal(evaluate(respy_obj), crit_val)
