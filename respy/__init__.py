"""This is the entrypoint to the respy package.

Include only imports which should be available using

.. code-block::

    import respy as rp

    rp.<func>

"""
import sys

import pytest

from respy.config import ROOT_DIR
from respy.interface import get_example_model  # noqa: F401
from respy.interface import get_parameter_constraints  # noqa: F401
from respy.likelihood import get_crit_func  # noqa: F401
from respy.method_of_simulated_moments import get_diag_weighting_matrix  # noqa: F401
from respy.method_of_simulated_moments import get_flat_moments  # noqa: F401
from respy.method_of_simulated_moments import get_msm_func  # noqa: F401
from respy.simulate import get_simulate_func  # noqa: F401
from respy.solve import get_solve_func  # noqa: F401
from respy.tests.random_model import add_noise_to_params  # noqa: F401

# We only maintain the code base for Python 3.6 and 3.7.
assert (3, 6) <= sys.version_info[:2] <= (3, 7)

__version__ = "2.0.0dev2"


def test(*args, **kwargs):
    """Run basic tests of the package."""
    pytest.main([str(ROOT_DIR), *args], **kwargs)
