
import numpy as np

import tempfile
import pytest
import os

@pytest.fixture()
def fresh_directory():
    os.chdir(tempfile.mkdtemp())

@pytest.fixture()
def set_seed():
    """ Fix the random seed for each test. 
    """
    np.random.seed(123)