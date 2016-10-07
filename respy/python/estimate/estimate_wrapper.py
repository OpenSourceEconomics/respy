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
        self.x_optim_all_unscaled = None
        self.paras_fixed = None
        self.maxfun = np.inf

        # Updated attributes
        # TODO: The setup of the optim container is different thanin FORT.
        # Here all parametes are included, in FORT only free.
        self.x_optim_container = np.tile(np.nan, (27, 3))
        self.x_econ_container = np.tile(np.nan, (27, 3))
        self.crit_vals = np.tile(np.inf, 3)
        self.num_step = -1
        self.num_eval = 0

    def crit_func(self, x_optim_free_scaled, *args):
        """ This method serves as a wrapper around the alternative
        implementations of the criterion function.
        """
        # Evaluate criterion function
        precond_matrix = self.precond_matrix

        from respy.python.shared.shared_auxiliary import cholesky_to_coeffs

        from numpy.linalg import pinv
        x_optim_free_unscaled = np.dot(pinv(precond_matrix), x_optim_free_scaled)

        x_optim_all_unscaled = self._construct_all_current_values(
            x_optim_free_unscaled)
        fval = pyth_criterion(x_optim_all_unscaled, *args)

        shocks_cholesky, _ = get_cholesky(x_optim_all_unscaled, 0)

        shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)

        # Identify events
        is_start = (self.num_eval == 0)
        is_step = (self.crit_vals[1] > fval)

        # Update class attributes
        if is_start:
            self.crit_vals[0] = fval
            self.x_optim_container[:, 0] = x_optim_all_unscaled
            self.x_econ_container[:17, 0] = x_optim_all_unscaled[:17]
            self.x_econ_container[17:, 0] = shocks_coeffs

        if is_step:
            self.num_step += 1
            self.crit_vals[1] = fval
            self.x_optim_container[:, 1] = x_optim_all_unscaled
            self.x_econ_container[:17, 1] = x_optim_all_unscaled[:17]
            self.x_econ_container[17:, 1] = shocks_coeffs

        if True:
            self.num_eval += 1
            self.crit_vals[2] = fval
            self.x_optim_container[:, 2] = x_optim_all_unscaled
            self.x_econ_container[:17, 2] = x_optim_all_unscaled[:17]
            self.x_econ_container[17:, 2] = shocks_coeffs

        if not hasattr(self, 'is_scaling'):
            # Record the progress of the estimation.
            record_estimation_eval(self, fval)

            # This is only used to determine whether a stabilization of the
            # Cholesky matrix is required.
            # TODO: I also need cholesky earlier
            _, info = get_cholesky(x_optim_all_unscaled, 0)
            if info != 0:
                record_warning(4)

            # Enforce a maximum number of function evaluations
            if self.maxfun == self.num_eval:
                raise MaxfunError

        # Finishing
        return fval

    def _construct_all_current_values(self, x_optim_free_unscaled):
        """ Construct the full set of current values.
        """
        x_optim_all_unscaled_start = self.x_optim_all_unscaled_start
        paras_fixed = self.paras_fixed

        x_optim_all_unscaled = np.tile(np.nan, 27)
        j = 0
        for i in range(27):
            if paras_fixed[i]:
                x_optim_all_unscaled[i] = x_optim_all_unscaled_start[i].copy()
            else:
                x_optim_all_unscaled[i] = x_optim_free_unscaled[j].copy()
                j += 1

        return x_optim_all_unscaled


class MaxfunError(Exception):
    """ This custom-error class allows to enforce the MAXFUN restriction
    independent of the optimizer used.
    """
    pass
