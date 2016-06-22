
import pickle as pkl

import os


import numpy as np


 
if True:
    assert os.system('./waf distclean; ./waf configure build --debug') == 0


from f2py_library import *


task = 'check'

###############################################################################
#   ŃEWUOA
###############################################################################
print(' ' + task + ' NEWUOA ...\n')
np.random.seed(123)

if task == 'check': 
    rslt = pkl.load(open('regresion_vault_newuoa.pkl', 'rb'))
elif task == 'create':
    rslt = []

for i in range(1000):

    dim = np.random.randint(2, 10)    

    p_start = np.random.uniform(-0.0, 0.1, size=dim)

    maxfun = np.random.randint(500, 20000)
    rhobeg= np.random.uniform(0.0001, 0.1, size=dim)
    rhoend=np.random.uniform(0.0, 0.99) * rhobeg
    npt=np.random.randint(5, 100)

    fval, p_final = f2py_newuoa(p_start, maxfun, rhobeg, rhoend, npt, dim)

    if task == 'check':
        np.testing.assert_almost_equal(rslt[i], [fval] + p_final.tolist())
    else:
        rslt += [[fval] + p_final.tolist()]

if task == 'create':
    pkl.dump(rslt, open('regresion_vault_newuoa.pkl', 'wb'))

###############################################################################
#   BFGS
###############################################################################
print(' ' + task + ' BFGS ...\n')
np.random.seed(123)

if task == 'check': 
    rslt = pkl.load(open('regresion_vault_bfgs.pkl', 'rb'))
elif task == 'create':
    rslt = []

for i in range(1000):

    dim = np.random.randint(2, 10)    


    p_start = np.random.uniform(-0.0, 0.1, size=dim)
    gtol = np.random.uniform(0.0001, 0.01)
    maxiter = np.random.randint(10, 200)
    stpmx = np.random.randint(50, 150)
    

    fval, p_final = f2py_bfgs(p_start, gtol, maxiter, stpmx, dim)


    if task == 'check':
        np.testing.assert_almost_equal(rslt[i], [fval] + p_final.tolist())
    else:
        rslt += [[fval] + p_final.tolist()]

if task == 'create':
    pkl.dump(rslt, open('regresion_vault_bfgs.pkl', 'wb'))
