# standard library
import pytest
import sys
import os

# project library
from respy.estimate import estimate
from respy.evaluate import evaluate
from respy.simulate import simulate
from respy.process import process
from respy.solve import solve
from respy.read import read

# Check for Python 3
if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')


""" Testing functions
"""


def test():
    """ Run nose tester for the package.
    """

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)
    pytest.main()
    os.chdir(current_directory)
