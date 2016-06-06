
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

    fval, p_final = f2py_newuoa(p_start, dim)

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

    fval, p_final = f2py_bfgs(p_start, dim)

    if task == 'check':
        np.testing.assert_almost_equal(rslt[i], [fval] + p_final.tolist())
    else:
        rslt += [[fval] + p_final.tolist()]

if task == 'create':
    pkl.dump(rslt, open('regresion_vault_bfgs.pkl', 'wb'))
