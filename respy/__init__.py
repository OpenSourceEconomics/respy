import warnings
import json
import os

# We want to set up some module-wide variables.
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

# We want to turn off the nuisance warnings while in production.
config = json.load(open(PACKAGE_DIR + '/.config'))
if not config['DEBUG']:
    warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import pytest

from respy.estimate import estimate
from respy.simulate import simulate
from respy.clsRespy import RespyCls

__version__ = '2.0.0.dev20'


def test(opt=None):
    """Run PYTEST for the package."""
    current_directory = os.getcwd()
    os.chdir(PACKAGE_DIR)

    if opt is None:
        opts = '-m"not slow"'
    else:
        opts = opt + ' -m"not slow"'

    pytest.main(opts)

    os.chdir(current_directory)
