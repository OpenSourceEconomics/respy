# Check for Python 3
import sys

#if not (sys.version_info[0] == 3):
#    raise AssertionError('Please use Python 3')

# Package structure
from robupy.read import read
from robupy.solve import solve
from robupy.simulate import simulate


""" Testing functions
"""
import os

def test():
    """ Run nose tester for the package.
    """
    base = os.getcwd()

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    os.chdir('tests')

    os.system('nosetests tests.py')

    os.chdir(base)


""" Set up logging.
"""
import logging

logging.captureWarnings(True)

formatter = logging.Formatter(' %(asctime)s     %(message)s \n',
                              datefmt='%I:%M:%S %p')



logger = logging.getLogger('ROBUPY_SOLVE')

handler = logging.FileHandler('logging.robupy.sol.log', mode='w', delay=True)

handler.setFormatter(formatter)

logger.setLevel(logging.INFO)

logger.addHandler(handler)


logger = logging.getLogger('ROBUPY_SIMULATE')

handler = logging.FileHandler('logging.robupy.sim.log', mode='w',
                              delay=True)

handler.setFormatter(formatter)

logger.setLevel(logging.INFO)

logger.addHandler(handler)
