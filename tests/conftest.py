
import numpy as np

import tempfile
import shutil
import pytest
import os
from material.auxiliary import compile_package



""" The following fixtures are called once per session.
"""

@pytest.fixture(scope='session')
def supply_resources():
    compile_package('--fortran --debug', True)


""" These fixtures are called before each test.
"""


@pytest.fixture(scope='function')
def set_seed():
    """ Fix the random seed for each test. 
    """
    np.random.seed(123)


@pytest.fixture(scope='function')
def fresh_directory():
    tmp_dir = tempfile.mkdtemp()
    os.chdir(tmp_dir)
