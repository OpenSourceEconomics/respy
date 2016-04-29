#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import os

# Build the package.
assert os.system('python setup.py build') == 0

# TOX automation
os.system('pip install tox')
assert os.system('tox -v') == 0

# Run PYTEST battery again for coverage statistic.
os.system('pip install pytest-cov==2.2.1')
assert os.system('py.test --cov=respy -v -s -m"(not slow)" -x') == 0
