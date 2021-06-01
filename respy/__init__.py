"""This is the entry-point to the respy package.

Include only imports which should be available using

.. code-block::

    import respy as rp

    rp.<func>

"""
import pytest

from respy.config import ROOT_DIR
from respy.interface import get_example_model  # noqa: F401
from respy.interface import get_parameter_constraints  # noqa: F401
from respy.likelihood import get_log_like_func  # noqa: F401
from respy.method_of_simulated_moments import get_diag_weighting_matrix  # noqa: F401
from respy.method_of_simulated_moments import get_flat_moments  # noqa: F401
from respy.method_of_simulated_moments import get_moment_errors_func  # noqa: F401
from respy.simulate import get_simulate_func  # noqa: F401
from respy.solve import get_solve_func  # noqa: F401
from respy.tests.random_model import add_noise_to_params  # noqa: F401


__all__ = [
    "get_example_model",
    "get_parameter_constraints",
    "get_solve_func",
    "get_simulate_func",
    "get_log_like_func",
    "get_moment_errors_func",
    "get_diag_weighting_matrix",
    "get_flat_moments",
    "add_noise_to_params",
]

__version__ = "2.1.0"


def test(*args, **kwargs):
    """Run basic tests of the package."""
    pytest.main([str(ROOT_DIR), *args], **kwargs)
