#!/usr/bin/env python
""" Script that prepare the virtual machine for the installation of SciPy and Pandas.
"""

# standard library
import os

# Installation.
packages = ['libblas-dev', 'liblapack-dev', 'gfortran']

for package in packages:

    os.system('sudo apt-get install -y ' + package)