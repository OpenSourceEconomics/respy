#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import os

# Get current directory.
CURRENT_DIR = os.getcwd()

# Get directory of ROBUPY package.
ROBUPY_DIR = os.path.abspath(os.path.dirname(__file__))
ROBUPY_DIR = ROBUPY_DIR.replace('/travis-ci', '/robupy')

# If the script is run on TRAVIS-CI, then I need to create a link to F2PY3. So
# far I was unable to figure out why that is the case.
if 'TRAVIS' in os.environ.keys():
    os.system('ln -sf /home/travis/virtualenv/python3.4.2/bin/f2py /home/travis/virtualenv/python3.4.2/bin/f2py3')

# Compile package as possible.
os.chdir(ROBUPY_DIR)
os.system('./waf distclean; ./waf configure build --fortran --debug')
os.chdir(CURRENT_DIR)

# Tests
os.system('nosetests --with-coverage --cover-package=robupy --exe')

os.system('coveralls')
