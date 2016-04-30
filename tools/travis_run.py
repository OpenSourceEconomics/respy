#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

# standard library
import subprocess as sp

# Ensure that most recent installation of PIP is available
sp.check_call('pip install --upgrade pip', shell=True)

# Download the most recent submission from PYPI and test it.
sp.check_call('pip install --no-binary respy -vvv respy', shell=True)
sp.check_call('python -c "import respy; respy.test()"', shell=True)
sp.check_call('pip uninstall respy', shell=True)

# Run PYTEST battery again on package in development mode. This ensure that the
# current implementation is working properly.
sp.check_call('pip install -e .', shell=True)
sp.check_call('pip install pytest-cov==2.2.1', shell=True)
sp.check_call('py.test --cov=respy -v -s -m"(not slow)" -x', shell=True)
