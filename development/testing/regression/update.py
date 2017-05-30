#!/usr/bin/env python
""" This script allows to update the regression tests.
"""

import subprocess
import shutil

from auxiliary_shared import cleanup

NUM_TESTS = 1000


# Initially we want to make sure that all the previous tests are running properly.
if True:
    cleanup()

    # This scratch file indicates that the required modification is done properly.
    open('.old.respy.scratch', 'w').close()
    cmd = './run_regression.py --request check ' + str(NUM_TESTS) + ' --strict'
    subprocess.check_call(cmd, shell=True)

# We create a new set of regression tests.
cleanup()
cmd = './run_regression.py --request create ' + str(NUM_TESTS)
subprocess.check_call(cmd, shell=True)

# These are subsequently copied into the test resources of the package.
src = 'regression_vault.respy.json'
dst = '../../../respy/tests/resources'
shutil.copy(src, dst)

# Just to be sure, we immediately check them again. This might fail if the random elements are
# not properly controlled for.
cleanup()
cmd = './run_regression.py --request check ' + str(NUM_TESTS) + ' --strict'
subprocess.check_call(cmd, shell=True)




