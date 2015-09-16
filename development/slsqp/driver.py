""" This module will allow me to get a bette handle on the optimization
problem I am facing.
"""
import shutil
import os
import sys
import glob

import numpy as np

sys.path.insert(0, os.environ['ROBUPY'])

from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

# project library
from robupy import *

from scipy.optimize import rosen_der, rosen

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



from slsqp import _minimize_slsqp

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



# Import 
import f2py_slsqp_debug as fort

    

# TODO NOSE test repeatedly, to get an automated count of failed tests...

# Ensure recomputability
np.random.seed(123)

for _ in range(1000):

    # Sample test case
    is_upgraded = np.random.choice([True, False])
    maxiter = np.random.random_integers(1, 100)
    num_dim = np.random.random_integers(2, 4)
    ftol = np.random.uniform(0.000000, 1e-5)
    x0 = np.random.normal(size=num_dim)

    # Test the upgraded FORTRAN version against the original code. This is 
    # expected to never fail.
    f_upgraded = fort.wrapper_slsqp_debug(x0, True, maxiter, ftol, num_dim)    
    f_original = fort.wrapper_slsqp_debug(x0, False, maxiter, ftol, num_dim)     

    np.testing.assert_array_equal(f_upgraded, f_original)

    # Test the FORTRAN codes against the PYTHON implementation. This is
    # expected to fail sometimes due to differences in precision between the
    # two implementations. In particular, as updating steps of the optimizer
    # are very sensitive to just small differences in the derivative
    # information.
    f = fort.wrapper_slsqp_robufort(x0, maxiter, ftol, num_dim)    
    py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter, 
            ftol=ftol)['x']

    np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)
