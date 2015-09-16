""" This module will allow me to get a bette handle on the optimization
problem I am facing.
"""
import shutil
import os
import sys
import glob

DEBUG_OPTIONS = ' -O2 -fimplicit-none  -Wall  -Wline-truncation ' \
                ' -Wcharacter-truncation  -Wsurprising  -Waliasing' \
                ' -Wimplicit-interface  -Wunused-parameter  -fwhole-file ' \
                ' -fcheck=all  -std=f2008  -pedantic  -fbacktrace'

import numpy as np

sys.path.insert(0, os.environ['ROBUPY'])

from numpy.testing.utils import assert_array_equal, assert_array_almost_equal

# project library
from robupy import *

from scipy.optimize import rosen_der, rosen

try:
    os.unlink('program')
except:
    pass

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
files = ['robufort_program_constants.f90', 'robufort_auxiliary.f90', 'slsqp_optmz_upgraded.f90']
for file_ in files:
    os.system('gfortran -c  -fPIC ' + file_)

os.system('gfortran -c   -fPIC --fixed-form slsqp_optmz.f')
os.system('ar crs libslsqp.a *.o *.mod')


module_files = glob.glob('*.mod')
for file_ in module_files:
    shutil.move(file_, 'include/')

shutil.move('libslsqp.a', 'lib/')

# Compile interface
os.system(
      'f2py3 -c -m  f2py_slsqp f2py_interface_slsqp.f90 -Iinclude -Llib '
        '-lslsqp')




#rslt = _minimize_slsqp(rosen, x0, jac = rosen_der)

#print('Python results')
#print(rslt['x'])

from slsqp import _minimize_slsqp
import f2py_slsqp as fort

np.random.seed(123)

for _ in range(10):

    ftol = np.random.uniform(0.000000, 1e-5)
    maxiter = np.random.random_integers(1, 100)
    num_dim = np.random.random_integers(2, 4)
    x0 = np.random.normal(size=num_dim)

    f90 = fort.wrapper_slsqp_upgraded(x0, maxiter, ftol, num_dim)

    f = fort.wrapper_slsqp_original(x0, maxiter, ftol, num_dim)
    py = _minimize_slsqp(rosen, x0, jac=rosen_der,
            maxiter=maxiter, ftol=ftol)['x']

    print(max(f90 - f))
    np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)
    np.testing.assert_array_equal(f, f90)