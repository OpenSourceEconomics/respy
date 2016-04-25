# standard library
import pytest
import sys
import os

# project library
from robupy.estimate import estimate
from robupy.evaluate import evaluate
from robupy.simulate import simulate
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

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)
    pytest.main()
    os.chdir(current_directory)
