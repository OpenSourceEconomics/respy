#!/usr/bin/env python
""" Script that prepare the virtual machine for the installation of SciPy
and Pandas.
"""

# standard library
import os

# Make sure that all available
os.system('sudo apt-get update -qq')

# Additional system-wide software.
packages = ['build-essential', 'gfortran', 'python3-pip', 'python-pip', 'git',
            'libblas-dev', 'libatlas-base-dev', 'liblapack-dev',
            'libyaml-cpp-dev', 'cython3', 'python-dev', 'python3-dev',
            'libevent-dev', 'python3-numpy']

for package in packages:
    os.system('sudo apt-get install -y ' + package)
