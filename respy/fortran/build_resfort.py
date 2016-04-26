""" This module contains all functions required to build the ROBUFORT
executable. Tedious modifications of the original source code is required to
get the maximum performance. The main component is an iterative inlining of
all functionality from the RESFORT library. This allows to maintain a
well-organized code without any loss of performance.
"""

# standard library
import shutil
import os


def resfort_build(self):
    """ Building the ROBUFORT executable for high speed execution.
    """
    # Compilation of executable
    current_directory = os.getcwd()

    compiler_options = self.env.compiler_options

    path = self.env.project_paths['RESPY']

    os.chdir(path + '/fortran')

    # Compile resfort file according to selected options.
    cmd = 'gfortran ' + compiler_options + ' -o resfort ' \
          'shared/shared_constants.f90 shared/shared_auxiliary.f90 ' \
          'resfort_slsqp.f ' \
          'solve/solve_auxiliary.f90 ' \
          'solve/solve_fortran.f90  evaluate/evaluate_auxiliary.f90 ' \
          'evaluate/evaluate_fortran.f90 estimate/estimate_auxiliary.f90 ' \
          'simulate/simulate_fortran.f90 ' \
          'resfort.f90 -L/usr/lib/lapack -llapack'

    os.system(cmd)

    # Maintaining directory structure.
    shutil.move('resfort', 'bin/resfort')

    os.chdir(current_directory)
