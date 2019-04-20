from datetime import datetime

import numpy as np

from respy.python.estimate.estimate_python import pyth_criterion
from respy.python.record.record_estimation import record_estimation_eval
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_auxiliary import apply_scaling
from respy.python.shared.shared_auxiliary import check_early_termination
from respy.python.shared.shared_auxiliary import extract_cholesky


class OptimizationClass(object):
    """Manage the optimization of the criterion function.

    The class provides a unified interface for a host of alternative
    optimization algorithms.

    """

    def __init__(
        self, x_optim_all_unscaled_start, paras_fixed, precond_matrix, num_types
    ):

        self.attr = dict()

        # Constitutive attributes
        self.x_optim_all_unscaled_start = x_optim_all_unscaled_start
        self.num_paras = len(x_optim_all_unscaled_start)
        self.precond_matrix = precond_matrix
        self.paras_fixed = paras_fixed
        self.num_types = num_types
        self.maxfun = np.inf

        num_paras = len(x_optim_all_unscaled_start)
        # Updated attributes
        self.x_optim_container = np.full((num_paras, 3), np.nan)
        self.x_econ_container = np.full((num_paras, 3), np.nan)
        self.crit_vals = np.full(3, np.inf)
        self.num_step = -1
        self.num_eval = 0

    def crit_func(self, x_optim_free_scaled, *args):
        """Wrapper for different implementations of the criterion function."""
        start = datetime.now()
        precond_matrix = self.precond_matrix
        x_optim_all_unscaled = self._construct_all_current_values(
            apply_scaling(x_optim_free_scaled, precond_matrix, "undo")
        )
        fval = pyth_criterion(x_optim_all_unscaled, *args)

        # Don't record anything if evaluating the criterion function simply to
        # get the precondition matrix.
        if not hasattr(self, "is_scaling"):

            # Record the progress of the estimation.
            record_estimation_eval(self, fval, x_optim_all_unscaled, start)

            # This is only used to determine whether a stabilization of the
            # Cholesky matrix is required.
            _, info = extract_cholesky(x_optim_all_unscaled, 0)
            if info != 0:
                record_warning(4)

            # Enforce a maximum number of function evaluations.
            check_early_termination(self.maxfun, self.num_eval)

        # Finishing
        return fval

    def _construct_all_current_values(self, x_optim_free_unscaled):
        """Construct the full set of current values."""
        x_optim_all_unscaled_start = self.x_optim_all_unscaled_start
        paras_fixed = self.paras_fixed
        num_paras = self.num_paras

        x_optim_all_unscaled = np.full(num_paras, np.nan)
        j = 0
        for i in range(num_paras):
            if paras_fixed[i]:
                x_optim_all_unscaled[i] = x_optim_all_unscaled_start[i].copy()
            else:
                x_optim_all_unscaled[i] = x_optim_free_unscaled[j].copy()
                j += 1

        return x_optim_all_unscaled
