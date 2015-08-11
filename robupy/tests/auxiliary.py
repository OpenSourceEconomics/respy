""" Auxiliary functions for development test suite.
"""

# standard library
import os

''' Auxiliary functions.
'''


def compile_package(which):
    """ Compile toolbox
    """
    # Antibugging
    assert (which in ['fast', 'slow'])

    # Auxiliary objects
    package_dir = os.environ['ROBUPY'] + '/robupy'
    tests_dir = os.getcwd()

    # Compile package
    os.chdir(package_dir)

    os.system('./waf distclean > /dev/null 2>&1')

    cmd = './waf configure build'

    if which == 'fast':
        cmd += ' --fast'

    cmd += ' > /dev/null 2>&1'

    os.system(cmd)

    os.chdir(tests_dir)
