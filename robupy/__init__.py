
# standard library
import logging
import pytest
import sys

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


""" Testing functions
"""


def test():
    """ Run nose tester for the package.
    """

    pytest.main('--cov=robupy -v')



""" Set up logging.
"""

logging.captureWarnings(True)

