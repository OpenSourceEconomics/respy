import respy

import subprocess
import sys
import os

cwd = os.getcwd()
if len(sys.argv) > 1:
    os.chdir('../../../respy')
    os.system('git clean -d -f; rm -rf .bld; ./waf configure build --debug')

os.chdir(cwd)


respy_obj = respy.RespyCls('test.respy.ini')
respy.estimate(respy_obj)
