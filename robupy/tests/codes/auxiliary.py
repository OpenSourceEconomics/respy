""" Auxiliary functions for development test suite.
"""

# standard library
import numpy as np

import fnmatch
import shutil
import glob
import os

# ROBUPY import
from robupy.shared.constants import ROOT_DIR
from robupy.solve.solve_auxiliary import pyth_create_state_space
from robupy import read

''' Auxiliary functions.
'''


def distribute_model_description(robupy_obj, *args):
    """ This function distributes the model description.
    """
    # TODO: I create a similar method as part of the robupy package. Mabe we
    # can replace this one?

    ret = []

    for arg in args:
        ret.append(robupy_obj.get_attr(arg))

    return ret


def write_interpolation_grid(file_name):
    """ Write out an interpolation grid that can be used across
    implementations.
    """
    # Process relevant initialization file
    robupy_obj = read(file_name)

    # Distribute class attribute
    num_periods, num_points, edu_start, edu_max, min_idx = \
        distribute_model_description(robupy_obj,
            'num_periods', 'num_points', 'edu_start', 'edu_max', 'min_idx')

    # Determine maximum number of states
    _, states_number_period, _, max_states_period = \
        pyth_create_state_space(num_periods, edu_start, edu_max, min_idx)

    # Initialize container
    booleans = np.tile(True, (max_states_period, num_periods))

    # Iterate over all periods
    for period in range(num_periods):

        # Construct auxiliary objects
        num_states = states_number_period[period]
        any_interpolation = (num_states - num_points) > 0

        # Check applicability
        if not any_interpolation:
            continue

        # Draw points for interpolation
        indicators = np.random.choice(range(num_states),
                            size=num_states - num_points, replace=False)

        # Replace indicators
        for i in range(num_states):
            if i in indicators:
                booleans[i, period] = False

    # Write out to file
    np.savetxt('interpolation.txt', booleans, fmt='%s')

    # Some information that is useful elsewhere.
    return max_states_period


def write_draws(num_periods, max_draws):
    """ Write out draws to potentially align the different
    implementations of the model. Note that num draws has to be less or equal
    to the largest number of requested random deviates.
    """
    # Draw standard deviates
    draws_standard = np.random.multivariate_normal(np.zeros(4),
        np.identity(4), (num_periods, max_draws))

    # Write to file to they can be read in by the different implementations.
    with open('draws.txt', 'w') as file_:
        for period in range(num_periods):
            for i in range(max_draws):
                line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}\n'.format(
                    *draws_standard[period, i, :])
                file_.write(line)


def build_robupy_package(is_hidden):
    """ Compile toolbox
    """
    # Auxiliary objects
    package_dir = ROOT_DIR
    tests_dir = os.getcwd()

    # Compile package
    os.chdir(package_dir)

    for i in range(1):

        os.system('./waf distclean > /dev/null 2>&1')

        cmd = './waf configure build --debug '

        if is_hidden:
            cmd += ' > /dev/null 2>&1'

        os.system(cmd)

    os.chdir(tests_dir)


def build_testing_library(is_hidden):
    """ Build the F2PY testing interface for testing.f
    """
    tmp_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists('build'):
        shutil.rmtree('build')

    os.mkdir('build')
    os.chdir('build')

    # Get all sources from the FORTRAN package.
    files = glob.glob('../../../fortran/*.f*')

    for file_ in files:
        shutil.copy(file_, '.')

    shutil.copy('../robufort_testing.f90', '.')
    shutil.copy('../f2py_interface_testing.f90', '.')

    # Build static library
    compiler_options = '-O3 -fpic'

    files = ['robufort_constants.f90', 'robufort_auxiliary.f90',
             'robufort_slsqp.f', 'robufort_emax.f90', 'robufort_risk.f90',
             'robufort_ambiguity.f90', 'robufort_library.f90',
             'robufort_testing.f90']

    for file_ in files:

        cmd = 'gfortran ' + compiler_options + ' -c ' + file_

        if is_hidden:
            cmd += ' > /dev/null 2>&1'
        os.system(cmd)

    os.system('ar crs libfort_testing.a *.o *.mod')

    # Prepare directory structure
    for dir_ in ['include', 'lib']:

        if os.path.exists(dir_):
            shutil.rmtree(dir_)

        os.makedirs(dir_)

    # Finalize static library
    module_files = glob.glob('*.mod')
    for file_ in module_files:
        shutil.move(file_, 'include/')
    shutil.move('libfort_testing.a', 'lib/')

    # Build interface
    cmd = 'f2py3 -c -m  f2py_testing f2py_interface_testing.f90 -Iinclude ' \
          ' -Llib -lfort_testing -L/usr/lib/lapack -llapack'

    if is_hidden:
        cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    lib_name = glob.glob('f2py_testing.*.so')[0]

    if not os.path.exists('../../lib/'):
        os.mkdir('../../lib/')

    shutil.copy(lib_name, '../../lib/')
    os.chdir('../')
    shutil.rmtree('build')
    # Finish
    os.chdir(tmp_dir)


def cleanup_robupy_package(is_build=False):
    """ This function deletes all nuisance files from the package
    """

    current_directory = os.getcwd()

    file_dir = os.path.realpath(__file__)
    package_dir = os.path.dirname(file_dir.replace('/tests/codes', ''))

    os.chdir(package_dir)

    # Collect all candidates files and directories.
    matches = []
    for root, dirnames, filenames in os.walk('.'):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '.*'):
            matches.append(os.path.join(root, filename))
        for dirname in fnmatch.filter(dirnames, '*'):
            matches.append(os.path.join(root, dirname))

    # Remove all files, unless explicitly to be saved.
    for match in matches:

        # If called as part of a build process, these temporary directories
        # are required.
        if is_build and (('.waf' in match) or ('.bld' in match)):
                continue

        # Explicit treatment for files.
        if os.path.isfile(match):
            if ('.py' in match) or ('.f' in match) or ('.f90' in match):
                continue

            # Keep README files for GITHUB
            if '.md' in match:
                continue

            # Keep files for build process
            if match in ['./waf', './wscript']:
                continue

            if match == './fortran/wscript':
                continue

            # Keep the initialization files for the regression tests.
            if 'test_' in match:
                continue

        else:

            if match == './fortran':
                continue

            if match == './python':
                continue

            if match == './python/py':
                continue

            if match == './python/f2py':
                continue

            if match == './tests':
                continue

            if match == './tests/codes':
                continue

            if match == './tests/resources':
                continue

            if match == './read':
                continue

            if match == './process':
                continue
            if match == './estimate':
                continue
            if match == './simulate':
                continue
            if match == './solve':
                continue
            if match == './evaluate':
                continue
            if match == './shared':
                continue

        # Remove remaining files and directories.. It is important to check
        # for the link first. Otherwise, isdir() evaluates to true but rmtree()
        # raises an error if applied to a link.
        if os.path.isfile(match) or os.path.islink(match):
            os.unlink(match)
        elif os.path.isdir(match):
            shutil.rmtree(match)

    os.chdir(current_directory)
