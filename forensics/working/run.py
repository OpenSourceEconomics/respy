#!/usr/bin/env python
""" This script compiles and executes the upgraded file that inspired the
RESPY package.
"""

import shutil
import os

# Compiler options. Note that the original codes are not robust enough to
# execute in debug mode.
DEBUG_OPTIONS = ' -O2  -Wall -Wline-truncation -Wcharacter-truncation ' \
    ' -Wsurprising  -Waliasing -Wimplicit-interface  -Wunused-parameter ' \
    ' -fwhole-file -fcheck=all  -fbacktrace -g -fmax-errors=1 ' \
    ' -ffpe-trap=invalid,zero'

PRODUCTION_OPTIONS = ' -O3'

# I rely on production options for this script, as I also run the estimation
# below.
OPTIONS = PRODUCTION_OPTIONS

# Some strings that show up repeatedly in compiler command.
MODULES = '_imsl_replacements.f90'
LAPACK = '-L/usr/lib/lapack -llapack'

# Compiling and calling executable for estimation.
cmd = ' gfortran ' + OPTIONS + ' -o dpml4a ' + MODULES + \
      ' dpml4a.f90 ' + LAPACK
os.system(cmd + '; ./dpml4a')

