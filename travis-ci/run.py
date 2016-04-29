#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import os

# If the script is run on TRAVIS-CI, then I need to create a link to F2PY3. So
# far I was unable to figure out why that is the case.
if 'TRAVIS' in os.environ.keys():
    os.system('ln -sf /home/travis/virtualenv/python3.4.2/bin/f2py /home/travis/virtualenv/python3.4.2/bin/f2py3')

# Build the package.
assert os.system('python setup.py build') == 0

# TODO: BE REMOVED
os.system('pip install pytest-cov==2.2.1')
assert os.system('py.test --cov=respy -v -s -m"(not slow)" -x') == 0


print("STARTING TOX")
# TOX automation
os.system('pip install tox')
assert os.system('tox -v -e py35') == 0

# Run PYTEST battery again for coverage statistic.
os.system('pip install pytest-cov==2.2.1')
assert os.system('py.test --cov=respy -v -s -m"(not slow)" -x') == 0
