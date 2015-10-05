#!/usr/bin/env python
""" Clean directory.
"""

# standard library
import shutil


dirs = ['modules/__pycache__', 'rslts']

for dir_ in dirs:
    try:
        shutil.rmtree(dir_)
    except OSError:
        pass

