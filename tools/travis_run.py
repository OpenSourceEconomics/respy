#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import os

# Ensure that most recent installation of PIP is available
assert os.system('pip install --upgrade pip')

# Download the most recent submission from PYPI and test it.
assert os.system('pip install --no-binary respy -vvv respy')
assert os.system('python -c "import respy; respy.test()"')
assert os.system('pip uninstall respy')

# TOX automation
os.system('pip install tox')
assert os.system('tox -v') == 0

# Run PYTEST battery again on package in development mode. This ensure that the
# current implementation is working properly.
assert os.system('pip install -e .') == 0
os.system('pip install pytest-cov==2.2.1')
assert os.system('py.test --cov=respy -v -s -m"(not slow)" -x') == 0
