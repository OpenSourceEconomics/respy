#!/usr/bin/env python
""" This script creates and writes to disk a list of the input for numerous
regression tests. These are included in the test infrastructure.
"""

# standard library
import pickle as pkl
import numpy as np

for idx in [2, 3]:

    # doing execfile() on this file will alter the current interpreter's
    # environment so you can import libraries in the virtualenv
    if idx == 2:
        activate_this_file = "/home/peisenha/.envs/restudToolbox2/bin/activate_this.py"
    if idx == 3:
        activate_this_file = "/home/peisenha/.envs/restudToolbox/bin/activate_this.py"


    execfile(activate_this_file, dict(__file__=activate_this_file))

    import sys
    version = str(sys.version_info[0])

    print(version)

#
# if version == '2':
#     sys.path.insert(0, '../../../respy/tests')
#     from codes.random_init import generate_init
# else:
#     from respy.tests.codes.random_init import generate_init
#
# from respy.python.shared.shared_auxiliary import print_init_dict
#
# from respy.evaluate import evaluate
#
# from respy import RespyCls
# from respy import simulate
#
# np.random.seed(213)
#
# num_tests = 100
# fname = 'test_vault_' + version + '.respy.pkl'
#
# tests = []
# for _ in range(num_tests):
#
#     init_dict = generate_init(constraints=None)
#
#     respy_obj = RespyCls('test.respy.ini')
#
#     simulate(respy_obj)
#
#     crit_val = evaluate(respy_obj)
#
#     test = (init_dict, crit_val)
#
#     tests += [test]
#
#     pkl.dump(tests, open(fname, 'wb'))
#
# print('.. done with creation.')
#
# # Now we make sure that the tests will pass.
# tests = pkl.load(open(fname, 'rb'))
#
# for test in tests:
#
#     init_dict, crit_val = test
#
#     print_init_dict(init_dict)
#
#     respy_obj = RespyCls('test.respy.ini')
#
#     simulate(respy_obj)
#
#     np.testing.assert_almost_equal(evaluate(respy_obj), crit_val)
