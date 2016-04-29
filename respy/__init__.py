# standard library
import pytest

import os
import sys

# project library
from respy.estimate import estimate
from respy.simulate import simulate
from respy.clsRespy import RespyCls

def test():
    """ Run PYTEST for the package.
    """

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)
    pytest.main()
    os.chdir(current_directory)
