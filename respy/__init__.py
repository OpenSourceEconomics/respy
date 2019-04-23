import os
import sys
import warnings

import pytest

from respy.clsRespy import RespyCls  # noqa: F401
from respy.python.shared.shared_constants import IS_DEBUG
from respy.python.shared.shared_constants import ROOT_DIR

# We only maintain the code base for Python >= 3.6
assert sys.version_info[:2] >= (3, 6)


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
