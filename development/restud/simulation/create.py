""" This scripts starts the reproduction of the simulated samples from the
original paper using the RESTUD program and the ROBUPY package.
"""

# standard library
import os


for dir_ in ['dp3asim', 'robupy']:

    os.chdir(dir_)

    os.sytem('python create.py')

    os.chdir('../')