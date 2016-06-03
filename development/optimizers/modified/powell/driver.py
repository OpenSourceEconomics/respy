
import os


from f2py_library import *

import numpy as np

while True:

    dim = np.random.randint(2, 10)    

    p_start = np.random.uniform(-0.1, 0.1, size=dim)

    print(dim, p_start)
    fval, p_final = f2py_powell(p_start, dim)

    print(fval, p_final)

    np.testing.assert_almost_equal(0, fval)

