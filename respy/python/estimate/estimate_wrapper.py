""" This module serves as a wrapper for the two alternative criterion functions.
"""

# standard library
import numpy as np

import time

# project library
from respy.python.estimate.estimate_python import pyth_criterion


class OptimizationClass(object):
    """ This class manages all about the optimization of the criterion
    function. It provides a unified interface for a host of alternative
    optimization algorithms.
    """

    def __init__(self):

        self.attr = dict()

        # constitutive arguments
        self.attr['num_step'] = -1
        self.attr['num_eval'] = 0

        self.attr['value_step'] = np.inf
        self.maxfun = np.inf
        self.x_container = np.tile(np.nan, (28, 3))

        self.x_all_start = None
        self.paras_fixed = None

    def crit_func(self, x, *args):
        """ This method serves as a wrapper around the alternative
        implementations of the criterion function.
        """

        if self.attr['num_eval'] == 0:
            is_start = True
        else:
            is_start = False

        # Evaluate criterion function
        x_all_start = self.x_all_start
        paras_fixed = self.paras_fixed

        x_all_current = np.tile(np.nan, 26)
        j = 0
        for i in range(26):
            if paras_fixed[i]:
                x_all_current[i] = x_all_start[i].copy()
            else:
                x_all_current[i] = x[j].copy()
                j += 1

        fval = pyth_criterion(x_all_current, *args)

        is_step = (self.attr['value_step'] > fval)

        if True:
            self.attr['num_eval'] += 1
            info_current = np.concatenate(([self.attr['num_eval'], fval], x_all_current))
            self.x_container[:, 2] = info_current

        if is_start:
            info_start = np.concatenate(([0, fval], x_all_current))
            self.x_container[:, 0] = info_start

        if is_step:

            self.attr['num_step'] += 1
            self.attr['value_step'] = fval

            info_step = np.concatenate(([self.attr['num_step'], fval], x_all_current))
            self.x_container[:, 1] = info_step


            with open('est.respy.log', 'a') as out_file:
                fmt_ = ' {0:>4}' + ' ' * 25 + '{1:>6}\n'
                out_file.write(fmt_.format(*['Step', self.attr['num_step']]))
                out_file.write(fmt_.format(*['Eval', self.attr['num_eval']]))
                fmt_ = ' {0:>9} ' + '{1:25.15f}\n'
                out_file.write(fmt_.format(*['Criterion', fval]))
                fmt_ = ' {0:<9} {1:>25}\n'
                out_file.write(fmt_.format(*['Time', time.strftime("%H:%M:%S")]))
                fmt_ = ' {0:<9} {1:>25}\n\n'
                out_file.write(fmt_.format(*['Date', time.strftime("%d/%m/%Y")]))

        np.savetxt(open('est.respy.paras', 'wb'), self.x_container,
                   fmt='%45.15f')

        # Enforce a maximum number of function evaluations
        if self.maxfun == self.attr['num_eval']:
            raise MaxfunError

        # Finishing
        return fval


class MaxfunError(Exception):
    """ This custom-error class allows to enforce the MAXFUN restriction
    independent of the optimzer used.
    """
    pass
