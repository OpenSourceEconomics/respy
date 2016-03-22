
# standard library
import logging
import sys
import os

# Package structure
from robupy.estimate import estimate
from robupy.simulate import simulate
from robupy.evaluate import evaluate
from robupy.process import process
from robupy.solve import solve
from robupy.read import read

# Check for Python 3
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')


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

