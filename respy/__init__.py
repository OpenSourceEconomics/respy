import os

try:
    import pytest
except ImportError:
    pass

from respy.estimate import estimate
from respy.simulate import simulate
from respy.clsRespy import RespyCls


__version__ = '1.0.0'
__name__ = 'respy'


def test(opt=None):
    """ Run PYTEST for the package.
    """

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)

    if opt is None:
        opts = '-m"not slow"'
    else:
        opts = opt + ' -m"not slow"'

    pytest.main(opts)

    os.chdir(current_directory)
