""" This module serves as a wrapper for the two alternative criterion functions.
"""

# standard library
from scipy.optimize import fmin_powell
from scipy.optimize import fmin_bfgs

import numpy as np

import time
import os

# project library
from respy.python.estimate.estimate_python import pyth_wrapper, pyth_criterion


class OptimizationClass(object):
    """ This class manages all about the optimization of the criterion
    function. It provides a unified interface for a host of alternative
    optimization algorithms.
    """

    def __init__(self):

        self.attr = dict()

        # constitutive arguments
        self.attr['optimizer_options'] = None

        self.attr['optimizer_used'] = None

        self.attr['file_opt'] = None

        self.attr['version'] = None

        self.attr['maxiter'] = None

        self.attr['x_info'] = None

        self.attr['args'] = None

        # status attributes
        self.attr['is_locked'] = False

        self.attr['is_first'] = True

    def set_attr(self, key, value):
        """ Set attributes.
        """
        # Antibugging
        assert (not self.attr['is_locked'])
        assert self._check_key(key)

        # Finishing
        self.attr[key] = value

    def optimize(self, x0):
        """ Optimize criterion function.
        """
        # Distribute class attributes
        optimizer_used = self.attr['optimizer_used']

        maxiter = self.attr['maxiter']

        args = self.attr['args']

        # Special case where just an evaluation at the starting values is
        # requested is accounted for. Note, that the relevant value of the
        # criterion function is always the one indicated by the class
        # attribute and not the value returned by the optimization algorithm.
        if maxiter == 0:
            crit_val = self.crit_func(x0, *args)

            rslt = dict()
            rslt['x'] = x0
            rslt['fun'] = crit_val
            rslt['success'] = True
            rslt['message'] = 'Evaluation of criterion function at starting values.'

        else:
            if optimizer_used == 'SCIPY-BFGS':
                gtol, epsilon = self._options_distribute(optimizer_used)
                fmin_bfgs(self.crit_func, x0, args=args, gtol=gtol,
                    epsilon=epsilon, maxiter=maxiter, full_output=True,
                    disp=False)

            elif optimizer_used == 'SCIPY-POWELL':
                xtol, ftol, maxfun = self._options_distribute(optimizer_used)
                fmin_powell(self.crit_func, x0, args, xtol, ftol,
                    maxiter, maxfun, full_output=True, disp=0)

            else:
                raise NotImplementedError

        # Reset teh static variables
        pyth_wrapper.fval_step = np.inf
        pyth_wrapper.num_steps = -1
        pyth_criterion.num_evals = 0


    def _options_distribute(self, optimizer_used):
        """ Distribute the optimizer specific options.
        """
        # Distribute class attributes
        options = self.attr['optimizer_options']

        # Extract optimizer-specific options
        options_opt = options[optimizer_used]

        # Construct options
        opts = None
        if optimizer_used == 'SCIPY-POWELL':
            opts = (options_opt['xtol'], options_opt['ftol'])
            opts += (options_opt['maxfun'],)

        elif optimizer_used == 'SCIPY-BFGS':
            opts = (options_opt['gtol'], options_opt['epsilon'])

        # Finishing
        return opts

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (not self.attr['is_locked'])

        # Update status indicator
        self.attr['is_locked'] = True

        # Checks
        self._check_integrity_attributes()

    def unlock(self):
        """ Unlock class instance.
        """
        # Antibugging
        assert self.attr['is_locked']

        # Update status indicator
        self.attr['is_locked'] = False

    def _check_key(self, key):
        """ Check that key is present.
        """
        # Check presence
        assert (key in self.attr.keys())

        # Finishing.
        return True

    def _check_integrity_attributes(self):
        """ Check integrity of class instance. This testing is done the first
        time the class is locked and if the package is running in debug mode.
        """
        # Distribute class attributes
        optimizer_options = self.attr['optimizer_options']

        optimizer_used = self.attr['optimizer_used']

        maxiter = self.attr['maxiter']

        version = self.attr['version']

        # Check that the options for the requested optimizer are available.
        assert (optimizer_used in optimizer_options.keys())

        # Check that version is available
        assert (version in ['PYTHON'])

        # Check that the requested number of iterations is valid
        assert isinstance(maxiter, int)
        assert (maxiter >= 0)

    def crit_func(self, x_free_curre, *args):
        """ This method serves as a wrapper around the alternative
        implementations of the criterion function.
        """
        # Get all parameters for the current evaluation
        x_all_curre = self._get_all_parameters(x_free_curre)

        # Evaluate criterion function
        crit_val = pyth_wrapper(x_all_curre, *args)

        # Antibugging.
        assert np.isfinite(crit_val)

        # Finishing
        return crit_val

    def _get_all_parameters(self, x_free_curr):
        """ This method constructs the full set of optimization parameters
        relevant for the current evaluation.
        """
        # Distribute class attributes
        x_all_start, paras_fixed = self.attr['x_info']

        # Initialize objects
        x_all_curre = []

        # Construct the relevant parameters
        j = 0
        for i in range(26):
            if paras_fixed[i]:
                x_all_curre += [float(x_all_start[i])]
            else:
                x_all_curre += [float(x_free_curr[j])]
                j += 1

        x_all_curre = np.array(x_all_curre)

        # Antibugging
        assert np.all(np.isfinite(x_all_curre))

        # Finishing
        return x_all_curre
