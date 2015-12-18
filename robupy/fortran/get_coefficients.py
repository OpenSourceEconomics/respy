#!/usr/bin/env python
""" This script allows to postpone the development of the Pseudo-Inverse 
computations for now.
"""

# standard library
import statsmodels.api as sm
import numpy as np
import os

# Read in endogenous and exogenous data from FORTRAN subroutine. 
X = np.array(np.genfromtxt('exogenous_variables.robupy.txt'), ndmin=2)
Y = np.array(np.genfromtxt('endogenous_variables.robupy.txt'), ndmin=1)

# Fit the model and obtain resulting parameters
coeffs = sm.OLS(Y, X).fit().params

# Write to file which is read in by FORTRAN subroutine.
np.savetxt('coeffs.robupy.txt', coeffs, fmt='%15.10f')

# Cleanup
for which in ['exogenous_variables', 'endogenous_variables']: 
    os.unlink(which + '.robupy.txt')