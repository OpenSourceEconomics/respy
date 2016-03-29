""" This module serves as a wrapper for the two alternative criterion functions.
"""

# standard library
from scipy.optimize import minimize
import numpy as np

import shlex
import os

# project library
from robupy.estimate.estimate_python import pyth_criterion
from robupy.shared.auxiliary import HUGE_FLOAT


from robupy.fortran.f2py_library import f2py_criterion

class OptimizationClass(object):
    """ This class manages all about the optimization of the criterion
    function. It provides a unified interface for a host of alternative
    optimization algorithms.
    """

    def __init__(self):

        self.attr = dict()

        # constitutive arguments
        self.attr['args'] = None

        self.attr['version'] = None

        self.attr['options'] = None

        self.attr['optimizer'] = None

        self.attr['maxiter'] = None

        # status attributes
        self.attr['step_paras'] = None

        self.attr['is_locked'] = False

        self.attr['is_first'] = True

        self.attr['step_val'] = HUGE_FLOAT

        self.attr['num_iter'] = 0

    def set_attr(self, key, value):
        """ Set attributes.
        """
        # Antibugging
        assert (not self.attr['is_locked'])
        assert self._check_key(key)

        # Finishing
        self.attr[key] = value

    def _check_key(self, key):
        """ Check that key is present.
        """
        # Check presence
        assert (key in self.attr.keys())

        # Finishing.
        return True

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (not self.attr['is_locked'])


        #self._check_integrity_results()

        # Read in optimizer options
        self.attr['options'] = self._options_read()
        self._options_check()

        # Update status indicator
        self.attr['is_locked'] = True

        # Checks
        self._check_integrity_attributes()

    def _check_integrity_attributes(self):
        """ Check integrity of class instance. This testing is done the first
        time the class is locked and if the package is running in debug mode.
        """
        # Distribute class attributes
        optimizer = self.attr['optimizer']

        maxiter = self.attr['maxiter']

        options = self.attr['options']

        version = self.attr['version']

        # Check that the options for the requested optimizer are available.
        assert (optimizer in options.keys())

        # Check that version is available
        assert (version in ['F2PY', 'FORTRAN', 'PYTHON'])

        # Check that the requested number of iterations is valid
        assert isinstance(maxiter, int)
        assert (maxiter >= 0)

    def optimize(self, x0):
        """ Optimize criterion function.
        """
        # Distribute class attributes
        optimizer = self.attr['optimizer']

        maxiter = self.attr['maxiter']

        options = self.attr['options']

        args = self.attr['args']

        # Subsetting to relevant set of options
        options = options[optimizer]

        # Special case where just an evaluation at the starting values is
        # requested is accounted for. Note, that the relevant value of the
        # criterion function is always the one indicated by the class
        # attribute and not the value returned by the optimization algorithm.
        if maxiter == 0:
            crit_val = self._crit_func(x0, *args)

            rslt = dict()
            rslt['x'] = x0
            rslt['fun'] = crit_val
            rslt['success'] = True
            rslt['message'] = 'Evaluation of criterion function at starting values.'

            self.attr['step+val'] = crit_val

        else:
            if optimizer == 'SCIPY-BFGS':
                rslt = minimize(self._crit_func, x0, method='BFGS', args=args,
                    options=options)
            elif optimizer == 'SCIPY-POWELL':
                rslt = minimize(self._crit_func, x0, method='POWELL', args=args,
                    options=options)
            else:
                raise NotImplementedError

        # Finalizing results
        fmt_ = '{0:<10} {1:<25}\n'
        with open('optimization.robupy.log', 'a') as out_file:
            out_file.write('Final Report\n\n')
            out_file.write(fmt_.format('Success', str(rslt['success'])))
            out_file.write(fmt_.format('Message', rslt['message']))
            out_file.write(fmt_.format('Criterion', self.attr['step_val']))

        # Finishing
        return rslt['x'], rslt['fun']

    def _crit_func(self, x, *args):
        """ This method serves as a wrapper around the alternative
        implementations of the criterion function.
        """
        # Distribute class attributes
        is_first = self.attr['is_first']

        step_val = self.attr['step_val']

        version = self.attr['version']

        num_iter = self.attr['num_iter']

        # Evaluate criterion function
        if version == 'PYTHON':
           crit_val = pyth_criterion(x, *args)
        elif version in ['F2PY', 'FORTRAN']:
            crit_val = f2py_criterion(x, *args)
        else:
            raise NotImplementedError

        # Recording of starting information
        if is_first:
            np.savetxt(open('start_paras.robupy.log', 'wb'), x, fmt='%15.8f')
            if os.path.exists('optimization.robupy.log'):
                os.unlink('optimization.robupy.log')

        # Recording of information about each step.
        if crit_val < step_val:
            np.savetxt(open('steps_paras.robupy.log', 'wb'), x, fmt='%15.8f')
            with open('optimization.robupy.log', 'a') as out_file:
                str_ = '{0:<10} {1:<5}\n'.format('Iteration', int(num_iter))
                out_file.write(str_)
                str_ = '{0:<10} {1:<5}\n\n\n'.format('Criterion', crit_val)
                out_file.write(str_)

            # Update class attributes
            self.attr['step_paras'] = x
            self.attr['step_val'] = crit_val
            self.attr['num_iter'] = num_iter + 1
            self.attr['is_first'] = False

        # Finishing
        return crit_val

    def unlock(self):
        """ Unlock class instance.
        """
        # Antibugging
        assert self.attr['is_locked']

        # Update status indicator
        self.attr['is_locked'] = False

    @staticmethod
    def _options_read():

        # Initialization
        dict_ = {}

        for line in open('optimization.robupy.opt').readlines():

            # Split line
            list_ = shlex.split(line)

            # Determine special cases
            is_empty, is_keyword = _process_cases(list_)

            # Applicability
            if is_empty:
                continue

            # Prepare dictionary
            if is_keyword:
                keyword = list_[0]
                dict_[keyword] = {}
                continue

            # Process blocks of information
            dict_ = _process_block(list_, dict_, keyword)

        return dict_

    def _options_check(self):
        """ Check the options for the admissible optimizers.
        """
        # Check options for the SCIPY-BFGS algorithm
        options = self.attr['options']['SCIPY-BFGS']
        gtol, epsilon = options['gtol'], options['epsilon']

        assert isinstance(gtol, float)
        assert (gtol > 0)

        assert isinstance(epsilon, float)
        assert (epsilon > 0)

        # Check options for the SCIPY-POWELL algorithm
        options = self.attr['options']['SCIPY-POWELL']
        xtol, ftol = options['xtol'], options['ftol']

        assert isinstance(xtol, float)
        assert (xtol > 0)

        assert isinstance(ftol, float)
        assert (ftol > 0)


""" Auxiliary functions
"""


def _process_block(list_, dict_, keyword):
    """ This function processes most parts of the initialization file.
    """
    # Distribute information
    name, val = list_[0], list_[1]

    # Prepare container.
    if (name not in dict_[keyword].keys()) and (name in ['coeff']):
        dict_[keyword][name] = []

    # Type conversion
    val = float(val)

    # Collect information
    dict_[keyword][name] = val

    # Finishing.
    return dict_

def _process_cases(list_):
    """ Process cases and determine whether keyword or empty line.
    """
    # Antibugging
    assert (isinstance(list_, list))

    # Get information
    is_empty = (len(list_) == 0)

    if not is_empty:
        is_keyword = list_[0].isupper()
    else:
        is_keyword = False

    # Antibugging
    assert (is_keyword in [True, False])
    assert (is_empty in [True, False])

    # Finishing
    return is_empty, is_keyword
