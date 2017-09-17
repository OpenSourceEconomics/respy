#!/usr/bin/env python
""" This script compiles and executes the upgraded file that inspired the RESPY
package.
"""
import linecache
import shlex
import os

import numpy as np

# Compiler options. Note that the original codes are not robust enough to execute in debug mode.
DEBUG_OPTIONS = ' -O2  -Wall -Wline-truncation -Wcharacter-truncation -Wsurprising  -Waliasing ' \
                '-Wimplicit-interface  -Wunused-parameter -fwhole-file -fcheck=all  -fbacktrace ' \
                '-g -fmax-errors=1 -ffpe-trap=invalid,zero'

PRODUCTION_OPTIONS = ' -O3'

# I rely on production options for this script, as I also run the estimation below.
OPTIONS = PRODUCTION_OPTIONS

# Some strings that show up repeatedly in compiler command.
MODULES = 'kw_imsl_replacements.f90 kw_test_additions.f90 '
LAPACK = '-L/usr/lib/lapack -llapack'

# Compiling and calling executable for estimation.
cmd = ' gfortran ' + OPTIONS + ' -o dpml4a ' + MODULES + ' dpml4a.f90 ' + LAPACK
os.system(cmd)

# Let me just fix a small regression test just to be sure ...
os.system('./dpml4a')
stat = float(shlex.split(linecache.getline('output1.txt', 65))[2])
np.testing.assert_equal(stat, -1019.77734375)
