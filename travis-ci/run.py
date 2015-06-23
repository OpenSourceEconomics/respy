#!/usr/bin/env python
""" Script that executes the testings for the Travis CI integration server.
"""

#standard library
import os

#Tests
os.system('nosetests --with-coverage --exe')