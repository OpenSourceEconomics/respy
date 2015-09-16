""" This module contains the testing infrastructure for the refactoring of the
SLSQP optimization algorithm.
"""

# standard library
from numpy.testing.utils import assert_array_almost_equal
from scipy.optimize import rosen_der, rosen

import numpy as np

import shutil
import glob
import sys
import os



# project library
sys.path.insert(0, os.environ['ROBUPY'])
from robupy import *


from slsqp import _minimize_slsqp

def compile_tools():
     # Cleanup function?
    for files in glob.glob('*.so'):
        os.unlink(files)


    # Cleaup
    for dir_ in ['include', 'lib']:
        try:
            shutil.rmtree(dir_)
        except:
            pass

        os.mkdir(dir_)




    # Create the SLSQP library
    files = ['robufort_program_constants.f90', 'robufort_auxiliary.f90', 'robufort_slsqp.f90']
    for file_ in files:
        os.system('gfortran -c  -fPIC ' + file_)

    os.system('gfortran -c   -fPIC --fixed-form original_slsqp.f')
    os.system('ar crs libslsqp_debug.a *.o *.mod')

    module_files = glob.glob('*.mod')
    for file_ in module_files:
        shutil.move(file_, 'include/')

    shutil.move('libslsqp_debug.a', 'lib/')

    # Compile interface
    os.system(
          'f2py3 -c -m  f2py_slsqp_debug f2py_interface_slsqp.f90 -Iinclude -Llib '
            '-lslsqp_debug')


def test_implementations():


    compile_tools()

    # Import
    import f2py_slsqp_debug as fort



    # TODO NOSE test repeatedly, to get an automated count of failed tests...

    # Ensure recomputability
    np.random.seed(345)

    for _ in range(1000):
        # Sample basic test case
        is_upgraded = np.random.choice([True, False])
        maxiter = np.random.random_integers(1, 100)
        num_dim = np.random.random_integers(2, 4)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=num_dim)

        # Add bounds
        shift = np.random.normal(size=2)**2
        bounds = np.vstack(( x0 - shift[0], x0 + shift[1])).T

        # Test the upgraded FORTRAN version against the original code. This is
        # expected to never fail.
        f_upgraded = fort.wrapper_slsqp_debug(x0, bounds, True, maxiter, ftol,
                                              num_dim)
        f_original = fort.wrapper_slsqp_debug(x0, bounds, False, maxiter, ftol,
                                              num_dim)

        np.testing.assert_array_equal(f_upgraded, f_original)

        # Test the FORTRAN codes against the PYTHON implementation. This is
        # expected to fail sometimes due to differences in precision between the
        # two implementations. In particular, as updating steps of the optimizer
        # are very sensitive to just small differences in the derivative
        # information.
        f = fort.wrapper_slsqp_debug(x0, bounds, is_upgraded, maxiter, ftol,
                                     num_dim)
        py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
                ftol=ftol, bounds=bounds)['x']

        #np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

#test_implementations()


       # Sample basic test case

np.random.seed(123)
is_upgraded = np.random.choice([True, False])
maxiter = 1000#np.random.random_integers(1, 100)
num_dim = np.random.random_integers(2, 4)
ftol = np.random.uniform(0.000000, 1e-5)
x0 = np.random.normal(size=num_dim)

# Add bounds
shift = np.random.normal(size=2)**2
bounds = np.vstack(( x0 - shift[0], x0 + shift[1])).T



def test_constraint_derivative(x):

    return np.ones(len(x))


def test_constraint(x):
    """ This constraint imposes that the sum of the parameters  needs to
        be larger than 10.
    """
    return np.sum(x) - 5


constraint = dict()

constraint['type'] = 'ineq'

constraint['fun'] = test_constraint

constraint['jac'] = test_constraint_derivative

constraint['args'] = ()

print(x0)
#py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
#               ftol=ftol, bounds=bounds, constraints=constraint)

compile_tools()

import f2py_slsqp_debug as fort

f = fort.wrapper_slsqp_debug(x0, bounds, is_upgraded, maxiter, ftol,
                                     num_dim)
#print(py)