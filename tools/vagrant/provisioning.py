#!/usr/bin/env python
""" This script provisions the virtual machine for the project.
"""

# standard library
import subprocess

# Required packages
apt_packages = ['python-pip', 'gfortran', 'libatlas-dev', 'libatlas-base-dev']
apt_packages += ['git']

# I manually install NUMPY, even though it is part of the RESPY requirements.
# Otherwise the install might fail due to an attempt to install SCIPY before
# NUMPY.
pyt_packages = ['numpy', 'respy']

# Housekeeping
subprocess.check_call(['apt-get', 'update'])
subprocess.check_call(['apt-get', 'upgrade'])

# Install some basic packages using the Advanced Packaging Tool.
for package in apt_packages:
    subprocess.check_call(['apt-get', 'install', '-y', package])

# Install packages using the PIP package manager.
for package in pyt_packages:
    cmd = ['pip', 'install', package]

    # I need to set the --no-binary flag to ensure a full compilation of the
    # FORTRAN resources. Otherwise, only sometimes, the libraries are not
    # compiled correctly.
    if 'respy' in package:
        cmd = ['pip', 'install', '--no-binary', 'respy', package]

    subprocess.check_call(cmd)

