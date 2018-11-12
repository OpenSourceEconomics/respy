import warnings
import sys
import os

import numpy as np
import pytest

from respy.python.shared.shared_constants import ROOT_DIR
from respy.python.shared.shared_constants import IS_DEBUG
from respy.clsRespy import RespyCls  # noqa: F401

# We only maintain the code base for modern Python.
major, minor = sys.version_info[:2]
np.testing.assert_equal(major == 3, True)
np.testing.assert_equal(minor >= 6, True)


# We want to turn off the nuisance warnings while in production.
if not IS_DEBUG:
    warnings.simplefilter(action="ignore", category=FutureWarning)

__version__ = "2.0.0.dev20"


def test(opt=None):
    """Run basic tests of the package."""
    current_directory = os.getcwd()
    os.chdir(ROOT_DIR)
    pytest.main(opt)
    os.chdir(current_directory)
