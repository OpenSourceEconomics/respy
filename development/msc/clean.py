#!/usr/bin/env python

""" Cleanup of directories.
"""

# standard library
import fnmatch
import shutil
import os

# module-wide variables
PROJECT_DIR = os.environ['ROBUPY']


os.system('git clean -d -f')
