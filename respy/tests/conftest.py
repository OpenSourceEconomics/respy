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
ROOT_DIR = ROOT_DIR.replace('/respy/tests', '')
sys.path.insert(0, ROOT_DIR)

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

