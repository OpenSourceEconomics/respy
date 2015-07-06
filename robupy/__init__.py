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
