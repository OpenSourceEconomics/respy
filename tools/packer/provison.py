#!/usr/bin/env python
""" This script provisions the virtual machine for the project.
"""

import subprocess
import os

# Required packages
apt_packages = ['python-dev', 'python-pip', 'python3-pip', 'gfortran', 'libatlas-dev']
apt_packages += ['libatlas-base-dev', 'git', 'libcr-dev', 'mpich', 'mpich-doc', 'python3-venv']

# I manually install NUMPY, even though it is part of the RESPY requirements.
# Otherwise the install might fail due to an attempt to install SCIPY before
# NUMPY.
pyt_packages = ['numpy', 'respy', 'virtualenvwrapper']

# Housekeeping
subprocess.check_call(['apt-get', 'update', '-y'])
subprocess.check_call(['apt-get', 'upgrade', '-y'])

# Install some basic packages using the Advanced Packaging Tool.
for package in apt_packages:
    subprocess.check_call(['apt-get', 'install', '-y', package])

# We need a more recent version of the PIP package manager.
subprocess.check_call(['pip', 'install', '--upgrade', 'pip'])

# Install packages using the PIP package manager.
for package in pyt_packages:
    cmd = ['pip', 'install', package]

    # I need to set the --no-binary flag to ensure a full compilation of the
    # FORTRAN resources. Otherwise, only sometimes, the libraries are not
    # compiled correctly.
    if 'respy' in package:
        cmd = ['pip', 'install', '--no-binary', 'respy', package]

    subprocess.check_call(cmd)

# Amend the shell configuration file to ease the work-flow of using virtual
# environments. 
with open('.profile', 'a') as outfile:
    outfile.write('export WORKON_HOME=$HOME/.envs \n')
    outfile.write('source /usr/local/bin/virtualenvwrapper.sh \n')
    
    outfile.write('\n\n export LC_ALL=C \n')
    
env_dir = os.environ['HOME'] + '/.envs'
cmd = ['pyvenv', env_dir + '/restudToolbox3', '--clear']
subprocess.check_call(cmd)

# Hook up to development repository.
#dirname = 'restudToolbox'
#os.mkdir(dirname), os.chdir(dirname)
#cmd = ['git', 'clone', 'https://github.com/restudToolbox/package.git']
#subprocess.check_call(cmd)

#os.chdir('package')
#python_exec = env_dir + '/restudToolbox3/bin/python'

# Fix permissions
#cmd = ['chwon', '-R', 'ubuntu:ubuntu', '*']
#subprocess.check_call(cmd)

