
# standard library
import logging
import atexit
import glob
import sys
import os

# Package structure
from robupy.estimate.estimate import estimate
from robupy.simulate.simulate import simulate
from robupy.evaluate.evaluate import evaluate
from robupy.process.process import process
from robupy.solve.solve import solve
from robupy.read.read import read

# Check for Python 3
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')


"""
"""
def cleanup_at_exit():
    """ Cleanup once call is finished.
    """
    for file_ in glob.glob('.*.scratch'):
        os.unlink(file_)

#atexit.register(cleanup_at_exit)

""" Testing functions
"""


def test():
    """ Run nose tester for the package.
    """
    base = os.getcwd()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    os.chdir('tests')

    os.system('nosetests tests.py')

    os.chdir(base)


""" Set up logging.
"""

logging.captureWarnings(True)

