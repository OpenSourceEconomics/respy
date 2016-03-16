""" This module provides the fixtures for the PYTEST runs.
"""
# standard library
import numpy as np

import tempfile
import pytest
import shutil
import os

# testing codes
from codes.auxiliary import build_f2py_testing
from codes.auxiliary import compile_package

# ROOT DIRECTORY
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/robupy/tests', '')

""" The following fixtures are called once per session.
"""


@pytest.fixture(scope='session')
def supply_resources(request):
    """ This fixture ensures that the compiled libraries are all available.
    """
    # Required compilations to make the F2PY and FORTRAN interfaces available.
    compile_package('--fortran --debug', True)

    # TODO: Better explanation.
    # This the shared robupy/tests
    build_f2py_testing(True)

    # Teardown of fixture after session is completed.
    def cleanup():
        shutil.rmtree(ROOT_DIR + '/robupy/tests/lib')

    request.addfinalizer(cleanup)

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

