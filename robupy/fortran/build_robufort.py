""" This module contains all functions required to build the ROBUFORT
executable. Tedious modifications of the original source code is required to
get the maximum performance. The main component is an iterative inlining of
all functionality from the ROBUFORT library. This allows to maintain a
well-organized code without any loss of performance.
"""

# standard library
import shutil
import os

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
    cmd = 'gfortran ' + compiler_options + ' -o robufort ' \
          'shared/shared_constants.f90 shared/shared_auxiliary.f90 ' \
          'robufort_slsqp.f  solve/solve_emax.f90 ' \
          'solve/solve_risk.f90 solve/solve_ambiguity.f90 solve/solve_auxiliary.f90 ' \
          'solve/solve.f90  evaluate/evaluate_auxiliary.f90 ' \
          'evaluate/evaluate.f90 estimate/estimate_auxiliary.f90 ' \
          'simulate/simulate.f90 ' \
          'robufort.f90 -L/usr/lib/lapack -llapack'

    os.system(cmd)

    # Maintaining directory structure.
    shutil.move('robufort', 'bin/robufort')

    os.chdir(current_directory)
