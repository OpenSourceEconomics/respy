
import os


from f2py_debug import *

import numpy as np
np.random.seed(123)

for _ in range(10):

    dim = np.random.randint(2, 10)    

    p_start = np.random.uniform(-0.1, 0.1, size=dim)

    print(dim, p_start)
    fval, p_final = f2py_bfgs(p_start, dim)

    print(fval, p_final)

    np.testing.assert_almost_equal(0, fval)

