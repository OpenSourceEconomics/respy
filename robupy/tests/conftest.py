""" This module provides the fixtures for the PYTEST runs.
"""
# standard library
import numpy as np

import tempfile
import pytest
import sys
import os

# ROOT DIRECTORY
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/robupy/tests', '')
sys.path.insert(0, ROOT_DIR)

# testing codes
from codes.auxiliary import cleanup_robupy_package
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package

""" The following fixtures are called once per session.
"""


def pytest_addoption(parser):
    """ Setup for PYTEST options.
    """
    parser.addoption("--versions", action="store",
        default=['PYTHON', 'F2PY', 'FORTRAN'],
        help="list of available versions", nargs='*')


@pytest.fixture
def versions(request):
    """ Processing of command line options.
    """
    arg_ = request.config.getoption("--versions")

    # Antibugging
    assert (isinstance(arg_, list))

    # Finishing
    return arg_


@pytest.fixture(scope='session')
def supply_resources(request):
    """ This fixture ensures that the compiled libraries are all available.
    """
    pass
""" The following fixtures are called before each test.
"""


@pytest.fixture(scope='function')
def set_seed():
    """ Each test is executed with the same random seed.
    """
    np.random.seed(1223)


@pytest.fixture(scope='function')
def fresh_directory():
    """ Each test is executed in a fresh directory.
    """
    os.chdir(tempfile.mkdtemp())

