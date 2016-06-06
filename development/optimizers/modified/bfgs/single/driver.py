
import pickle as pkl

import os


import numpy as np

np.random.seed(123)#

 
if True:
    assert os.system('./waf distclean; ./waf configure build --debug') == 0


from f2py_library import *

#rslt = []
rslt = pkl.load(open('regresion_vault.pkl', 'rb'))

for i in range(100):
    print('Testing ', i)
    dim = np.random.randint(2, 10)    

    p_start = np.random.uniform(-0.1, 0.1, size=dim)

    fval, p_final = f2py_bfgs(p_start, dim)

#    rslt += [[fval] + p_final.tolist()]
    np.testing.assert_almost_equal(rslt[i], [fval] + p_final.tolist())

# Store regression vault.
#pkl.dump(rslt, open('regresion_vault.pkl', 'wb'))