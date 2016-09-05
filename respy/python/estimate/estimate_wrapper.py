import numpy as np

from respy.python.record.record_estimation import record_estimation_eval
from respy.python.estimate.estimate_python import pyth_criterion
from respy.python.shared.shared_auxiliary import get_cholesky
from respy.python.record.record_warning import record_warning


class OptimizationClass(object):
    """ This class manages all about the optimization of the criterion
    function. It provides a unified interface for a host of alternative
    optimization algorithms.
    """

    def __init__(self):

        self.attr = dict()

        # Constitutive attributes
        self.x_all_start = None
        self.paras_fixed = None
        self.maxfun = np.inf

        # Updated attributes
        self.x_container = np.tile(np.nan, (27, 3))
        self.crit_vals = np.tile(np.inf, 3)
        self.num_step = -1
        self.num_eval = 0

    def crit_func(self, x, *args):
        """ This method serves as a wrapper around the alternative
        implementations of the criterion function.
        """
        # Evaluate criterion function
        x_all_current = self._construct_all_current_values(x)
        fval = pyth_criterion(x_all_current, *args)

        # Identify events
        is_start = (self.num_eval == 0)
        is_step = (self.crit_vals[1] > fval)

        # Update class attributes
        if is_start:
            self.crit_vals[0] = fval
            self.x_container[:, 0] = x_all_current

        if is_step:
            self.num_step += 1
            self.crit_vals[1] = fval
            self.x_container[:, 1] = x_all_current

        if True:
            self.num_eval += 1
            self.crit_vals[2] = fval
            self.x_container[:, 2] = x_all_current

        # Record the progress of the estimation.
        record_estimation_eval(self, fval)

        # This is only used to determine whether a stabilization of the
        # Cholesky matrix is required.
        _, info = get_cholesky(x_all_current, 0)
        if info != 0:
            record_warning(4)

        # Enforce a maximum number of function evaluations
        if self.maxfun == self.num_eval:
            raise MaxfunError

        # Finishing
        return fval

    def _construct_all_current_values(self, x):
        """ Construct the full set of current values.
        """
        x_all_start = self.x_all_start
        paras_fixed = self.paras_fixed

        x_all_current = np.tile(np.nan, 27)
        j = 0
        for i in range(27):
            if paras_fixed[i]:
                x_all_current[i] = x_all_start[i].copy()
            else:
                x_all_current[i] = x[j].copy()
                j += 1

        return x_all_current


class MaxfunError(Exception):
    """ This custom-error class allows to enforce the MAXFUN restriction
    independent of the optimizer used.
    """
    pass
