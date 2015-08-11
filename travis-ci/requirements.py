#!/usr/bin/env python
""" Script that prepare the virtual machine for the installation of SciPy
and Pandas.
"""

# standard library
import os

# Additional system-wide software.
packages = ['build-essential', 'gfortran', 'python3-pip', 'python-pip', 'git',
            'libblas-dev', 'libatlas-base-dev', 'liblapack-dev',
            'libyaml-cpp-dev', 'cython3', 'python-dev', 'python3-dev',
            'libevent-dev']

for package in packages:
    os.system('sudo apt-get install -y ' + package)
