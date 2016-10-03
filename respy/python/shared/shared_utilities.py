import numpy as np
import scipy


def spectral_condition_number(mat):

    svs = scipy.linalg.svd(mat)[1]
    cond = np.amax(abs(svs)) / np.amin(abs(svs))

    return cond
