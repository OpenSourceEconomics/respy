""" This module provides the fixtures for the PYTEST runs.
"""
# standard library
import numpy as np

import tempfile
import pytest
import shutil
import sys
import os

# ROOT DIRECTORY
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/robupy/tests', '')
sys.path.insert(0, ROOT_DIR)

# testing codes
from codes.auxiliary import build_testing_library
from codes.auxiliary import build_robupy_package

# ROBUPY packages
from robupy.auxiliary import cleanup_robupy_package

""" The following fixtures are called once per session.
"""


@pytest.fixture(scope='session')
def supply_resources(request):
    """ This fixture ensures that the compiled libraries are all available.
    """
    # Required compilations to make the F2PY and FORTRAN interfaces available.
    build_robupy_package(True)

    # There is small number of FORTRAN routines that are only used during
    # testing. These are collected in their own library.
    build_testing_library(True)

    # Teardown of fixture after session is completed.
    request.addfinalizer(cleanup_robupy_package)

""" The following fixtures are called before each test.
"""


@pytest.fixture(scope='function')
def set_seed():
    """ Each test is executed with the same random seed.
    """
    np.random.seed(123)


@pytest.fixture(scope='function')
def fresh_directory():
    """ Each test is executed in a fresh directory.
    """
    os.chdir(tempfile.mkdtemp())

