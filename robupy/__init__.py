# Check for Python 3
import sys

if not (sys.version_info[0] == 3):
    raise AssertionError('Please use Python 3')

# Package structure
from .read import read
from .solve import solve
from .simulate import simulate


""" Set up logging.
"""
import logging

logging.captureWarnings(True)

logger    = logging.getLogger('ROBUPY')

formatter = logging.Formatter(' %(asctime)s     %(message)s \n', datefmt =
'%I:%M:%S %p')

handler   = logging.FileHandler('logging.robupy.log', mode = 'w')

handler.setFormatter(formatter)

logger.setLevel(logging.INFO)

logger.addHandler(handler)
