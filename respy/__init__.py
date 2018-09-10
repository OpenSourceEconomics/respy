import warnings
import json
import sys
import os

# We want to set up some module-wide variables.
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

# We want to turn off the nuisance warnings while in production.
config = json.load(open(PACKAGE_DIR + '/.config'))
if not config['DEBUG']:
    warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pytest

# We only maintain the code base for modern Python.
major, minor = sys.version_info[:2]
np.testing.assert_equal(major == 3, True)
np.testing.assert_equal(minor >= 6, True)

from respy.clsRespy import RespyCls

__version__ = '2.0.0.dev20'


def test(opt=None):
    """Run PYTEST for the package."""
    current_directory = os.getcwd()
    os.chdir(PACKAGE_DIR)
    pytest.main(opt)
    os.chdir(current_directory)
