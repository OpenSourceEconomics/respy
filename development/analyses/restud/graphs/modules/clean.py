#!/usr/bin/env python
""" Clean directory.
"""

# standard library
import shutil
import os
import sys


dirs = ['__pycache__', 'rslts', 'modules/__pycache__']

for dir_ in dirs:
    try:
        shutil.rmtree(dir_)
    except OSError:
        pass

