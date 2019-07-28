"""This is the entrypoint to the respy package.

Include only imports which should be available using

.. code-block::

    import respy as rp

    rp.<func>

"""
import os
import sys

import pytest

from respy.config import ROOT_DIR
from respy.likelihood import get_crit_func  # noqa: F401
from respy.shared import get_example_model  # noqa: F401
from respy.simulate import get_simulate_func  # noqa: F401
from respy.solve import solve  # noqa: F401

# We only maintain the code base for Python >= 3.6.
assert sys.version_info[:2] >= (3, 6)

__version__ = "1.2.1"


def test(opt=None):
    """Run basic tests of the package."""
    current_directory = os.getcwd()
    os.chdir(ROOT_DIR)
    pytest.main(opt)
    os.chdir(current_directory)
