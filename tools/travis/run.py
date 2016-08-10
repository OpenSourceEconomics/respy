#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

import subprocess as sp
import numpy as np
import sys
import os

# Ensure that most recent installation of PIP is available
sp.check_call('pip install --upgrade pip', shell=True)

# Test the most recent PYPI submission. We need to switch to the directory to
# for the site-packages to avoid the wrong import. This only works for the
# more recent Python versions.
version = sys.version_info[:2]
if version in [(3, 4), (3, 5)]:
    cwd = os.getcwd()
    site_packages = np.__path__[0].replace('numpy', '')
    sp.check_call('pip install -vvv --no-binary respy respy', shell=True)
    os.chdir(site_packages)
    sp.check_call('python -c "import respy; respy.test()"', shell=True)
    sp.check_call('pip uninstall -y respy', shell=True)
    os.chdir(cwd)

# Run PYTEST battery again on package in development mode. This ensure that the
# current implementation is working properly.
sp.check_call('pip install -e .', shell=True)
sp.check_call('pip install pytest-cov==2.2.1', shell=True)
sp.check_call('py.test --cov=respy -v -s -x', shell=True)
