#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import pytest
import os

# If the script is run on TRAVIS-CI, then I need to create a link to F2PY3. So
# far I was unable to figure out why that is the case.
if 'TRAVIS' in os.environ.keys():
    os.system('ln -sf /home/travis/virtualenv/python3.4.2/bin/f2py /home/travis/virtualenv/python3.4.2/bin/f2py3')

# Build the package
os.chdir('respy')
os.system('./waf distclean; ./waf configure build --debug')
os.chdir('../')

# Run PYTEST battery, some tests are expected to fail due to small numerical
# differences between PYTHON and FORTRAN implementations.
os.system('pip install pytest-cov==2.2.1')
assert pytest.main('--cov=respy -v -s -m"(not slow)"')
