#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import os

# If the script is run on TRAVIS-CI, then I need to create a link to F2PY3. So
# far I was unable to figure out why that is the case. I am creating the link
# for both distributions so that the TOX setup works properly.
if 'TRAVIS' in os.environ.keys():
    os.system('ln -sf /home/travis/virtualenv/python3.4.2/bin/f2py /home/travis/virtualenv/python3.4.2/bin/f2py3')
    os.system('ln -sf /home/travis/virtualenv/python3.5.0/bin/f2py /home/travis/virtualenv/python3.5.0/bin/f2py3')

# Build the package.
assert os.system('python setup.py build') == 0

# Run PYTEST battery, some tests might fail due to small numerical
# differences between PYTHON and FORTRAN implementations.
os.system('pip install pytest-cov==2.2.1')
assert os.system('py.test --cov=respy -v -s -m"(not slow)" -x') == 0

# TOX automation for the development version
os.system('pip install tox')
assert os.system('tox -v') == 0
