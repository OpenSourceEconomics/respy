""" This module contains all functions required to build the ROBUFORT
executable. Tedious modifications of the original source code is required to
get the maximum performance. The main component is an iterative inlining of
all functionality from the ROBUFORT library. This allows to maintain a
well-organized code without any loss of performance.
"""

# standard library
import shutil
import shlex
import copy
import os

# module-wide variables
DEBUG_OPTIONS = ' -O2 -fimplicit-none  -Wall -Wline-truncation' \
                ' -Wcharacter-truncation  -Wsurprising  -Waliasing' \
                ' -Wimplicit-interface  -Wunused-parameter  -fwhole-file' \
                ' -fcheck=all  -fbacktrace '

PRODUCTION_OPTIONS = '-O3'


''' Main function
'''


def robufort_build(self, is_debug=False):
    """ Building the ROBUFORT executable for high speed execution.
    """
    # Compilation of executable for fastest performance
    current_directory = os.getcwd()

    path = self.env.project_paths['ROBUPY']

    os.chdir(path + '/fortran')

    # Set compilation options
    if is_debug:
        compiler_options = DEBUG_OPTIONS
    else:
        compiler_options = PRODUCTION_OPTIONS

    # Compile robufort file according to selected options.
    cmd = 'gfortran ' + compiler_options + ' -o robufort_risk ' \
          'robufort_constants.f90  robufort_auxiliary.f90 ' \
          'robufort_slsqp.f robufort_emax.f90 robufort_risk.f90  ' \
          'robufort_ambiguity.f90 robufort_library.f90 ' \
          'robufort.f90 -L/usr/lib/lapack -llapack'

    os.system(cmd)

    # There will be a separate executable for the risk and ambiguity case.
    # The separation is required for optimal performance in the case of
    # optimization.
    shutil.copy('robufort_risk', 'robufort_ambiguity')

    # Maintaining directory structure.
    for file_ in ['risk', 'ambiguity']:
        shutil.move('robufort_' + file_, 'bin/robufort_' + file_)

    os.chdir(current_directory)
