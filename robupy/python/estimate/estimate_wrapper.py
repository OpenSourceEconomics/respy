""" This module serves as a wrapper for the two alternative criterion functions.
"""

# standard library
from scipy.optimize import fmin_powell
from scipy.optimize import fmin_bfgs

import numpy as np

import shlex
import os

# project library
from robupy.python.estimate.estimate_auxiliary import dist_optim_paras
from robupy.python.estimate.estimate_auxiliary import process_block
from robupy.python.estimate.estimate_auxiliary import process_cases

from robupy.python.estimate.estimate_python import pyth_criterion

from robupy.python.shared.shared_auxiliary import HUGE_FLOAT

from robupy.fortran.f2py_library import f2py_criterion


class OptimizationClass(object):
    """ This class manages all about the optimization of the criterion
    function. It provides a unified interface for a host of alternative
    optimization algorithms.
    """

    def __init__(self):

        self.attr = dict()

        # constitutive arguments
        self.attr['optimizer'] = None

        self.attr['file_opt'] = None

        self.attr['version'] = None

        self.attr['options'] = None

        self.attr['maxiter'] = None

        self.attr['x_info'] = None

        self.attr['args'] = None

        # status attributes
        self.attr['value_steps'] = HUGE_FLOAT

        self.attr['value_start'] = HUGE_FLOAT

        self.attr['value_curre'] = HUGE_FLOAT

        self.attr['paras_steps'] = None

        self.attr['paras_curre'] = None

        self.attr['paras_start'] = None

        self.attr['is_locked'] = False

        self.attr['is_first'] = True

        self.attr['num_steps'] = 0

        self.attr['num_evals'] = 0

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
        optimizer = self.attr['optimizer']

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
            if optimizer == 'SCIPY-BFGS':
                gtol, eps = self._options_distribute(optimizer)
                rslt_opt = fmin_bfgs(self.crit_func, x0, args=args, gtol=gtol,
                    epsilon=eps, maxiter=maxiter, full_output=True, disp=False)

            elif optimizer == 'SCIPY-POWELL':
                xtol, ftol, maxfun = self._options_distribute(optimizer)
                rslt_opt = fmin_powell(self.crit_func, x0, args, xtol, ftol,
                    maxiter, maxfun, full_output=True)

            else:
                raise NotImplementedError

            # Process results to a common dictionary.
            rslt = self._results_distribute(rslt_opt, optimizer)

        # Finalizing results
        self._logging_final(rslt)

        # Finishing
        return rslt['x'], rslt['fun']

    def _results_distribute(self, rslt_opt, optimizer):
        """ Distribute results from the return from each of the optimizers.
        """
        # Initialize dictionary
        rslt = dict()
        rslt['x'] = None
        rslt['fun'] = None
        rslt['success'] = None
        rslt['message'] = None

        # Get the best parametrization and the corresponding value of the
        # criterion function.
        rslt['x'] = self.attr['paras_steps']
        rslt['fun'] = self.attr['value_steps']

        # Special treatments for each optimizer to extract message and
        # success indicator.
        if optimizer == 'SCIPY-POWELL':
            rslt['success'] = (rslt_opt[5] not in [1, 2])
            rslt['message'] = 'Optimization terminated successfully.'
            if rslt_opt[5] == 1:
                rslt['message'] = 'Maximum number of function evaluations.'
            elif rslt_opt[5] == 2:
                rslt['message'] = 'Maximum number of iterations.'

        elif optimizer == 'SCIPY-BFGS':
            rslt['success'] = (rslt_opt[5] not in [1, 2])
            rslt['message'] = 'Optimization terminated successfully.'
            if rslt_opt[5] == 1:
                rslt['message'] = 'Maximum number of iterations exceeded.'
            elif rslt_opt[5] == 2:
                rslt['message'] = 'Gradient and/or function calls not changing.'

        else:
            raise NotImplementedError

        # Finishing
        return rslt

    def _options_distribute(self, optimizer):
        """ Distribute the optimizer specific options.
        """
        # Distribute class attributes
        options = self.attr['options']

        # Extract optimizer-specific options
        options_opt = options[optimizer]

        # Construct options
        if optimizer == 'SCIPY-POWELL':
            opts = (options_opt['xtol'], options_opt['ftol'])
            opts += (options_opt['maxfun'],)

        elif optimizer == 'SCIPY-BFGS':
            opts = (options_opt['gtol'], options_opt['eps'])

        # Finishing
        return opts

    def lock(self):
        """ Lock class instance.
        """
        # Antibugging.
        assert (not self.attr['is_locked'])

        # Read in optimizer options
        self.attr['options'] = self._options_read()
        self._options_check()

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

    def crit_func(self, x_free_curre, *args):
        """ This method serves as a wrapper around the alternative
        implementations of the criterion function.
        """
        # Distribute class attributes
        version = self.attr['version']

        # Get all parameters for the current evaluation
        x_all_curre = self._get_all_parameters(x_free_curre)

        # Evaluate criterion function
        if version == 'PYTHON':
            crit_val = pyth_criterion(x_all_curre, *args)
        elif version in ['F2PY', 'FORTRAN']:
            crit_val = f2py_criterion(x_all_curre, *args)
        else:
            raise NotImplementedError

        # Antibugging.
        assert np.isfinite(crit_val)

        # Document progress
        self._logging_interim(x_all_curre, crit_val)

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
        for i in range(16):
            if paras_fixed[i]:
                x_all_curre += [float(x_all_start[i])]
            else:
                x_all_curre += [float(x_free_curr[j])]
                j += 1

        # Special treatment of SHOCKS_COV
        if paras_fixed[16]:
            x_all_curre += x_all_start[16:].tolist()
        else:
            x_all_curre += x_free_curr[-10:].tolist()

        x_all_curre = np.array(x_all_curre)

        # Antibugging
        assert np.all(np.isfinite(x_all_curre))

        # Finishing
        return x_all_curre

    def _options_read(self):
        """ Read in the tuning parameters for the optimizers.
        """
        # Distribute class attributes
        file_opt = self.attr['file_opt']

        # Initialization
        dict_, keyword = {}, None

        with open(file_opt) as in_file:

            for line in in_file.readlines():

                # Split line
                list_ = shlex.split(line)

                # Determine special cases
                is_empty, is_keyword = process_cases(list_)

                # Applicability
                if is_empty:
                    continue

                # Prepare dictionary
                if is_keyword:
                    keyword = list_[0]
                    dict_[keyword] = {}
                    continue

                # Process blocks of information
                dict_ = process_block(list_, dict_, keyword)

        return dict_

    def _options_check(self):
        """ Check the options for the admissible optimizers.
        """
        # Check options for the SCIPY-BFGS algorithm
        options = self.attr['options']['SCIPY-BFGS']
        gtol, epsilon = options['gtol'], options['eps']

        assert isinstance(gtol, float)
        assert (gtol > 0)

        assert isinstance(epsilon, float)
        assert (epsilon > 0)

        # Check options for the SCIPY-POWELL algorithm
        options = self.attr['options']['SCIPY-POWELL']
        xtol, ftol = options['xtol'], options['ftol']
        maxfun = options['maxfun']

        assert isinstance(maxfun, int)
        assert (maxfun > 0)

        assert isinstance(xtol, float)
        assert (xtol > 0)

        assert isinstance(ftol, float)
        assert (ftol > 0)

    def _logging_interim(self, x, crit_val):
        """ This method write out some information during the optimization.
        """
        # Recording of current evaluation
        self.attr['num_evals'] += 1
        self.attr['value_curre'] = crit_val
        self.attr['paras_curre'] = x

        # Distribute class attributes
        paras_curre = self.attr['paras_curre']

        value_curre = self.attr['value_curre']
        value_steps = self.attr['value_steps']

        num_steps = self.attr['num_steps']
        num_evals = self.attr['num_evals']

        is_first = self.attr['is_first']

        np.savetxt(open('paras_curre.robupy.log', 'wb'), x, fmt='%15.8f')

        # Recording of starting information
        if is_first:
            self.attr['value_start'] = crit_val
            self.attr['paras_start'] = x
            np.savetxt(open('paras_start.robupy.log', 'wb'), x, fmt='%15.8f')
            if os.path.exists('optimization.robupy.log'):
                os.unlink('optimization.robupy.log')

        paras_start = self.attr['paras_start']
        value_start = self.attr['value_start']

        # Recording of information about each step.
        if crit_val < value_steps:
            np.savetxt(open('paras_steps.robupy.log', 'wb'), x, fmt='%15.8f')
            with open('optimization.robupy.log', 'a') as out_file:
                str_ = '{0:<10} {1:<5}\n'.format('Iteration', int(num_steps))
                out_file.write(str_)
                str_ = '{0:<10} {1:<5}\n\n\n'.format('Criterion', crit_val)
                out_file.write(str_)

            # Update class attributes
            self.attr['paras_steps'] = x
            self.attr['value_steps'] = crit_val
            self.attr['num_steps'] = num_steps + 1
            self.attr['is_first'] = False

        paras_steps = self.attr['paras_steps']
        value_steps = self.attr['value_steps']

        # Write information to file.
        with open('optimization.robupy.info', 'w') as out_file:
            # Write out information about criterion function
            out_file.write('\n Criterion Function\n\n')
            fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
            out_file.write(fmt_.format(*['', 'Start', 'Step', 'Current']))
            fmt_ = '{0:>15}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n\n'
            paras = ['', value_start, value_steps, value_curre]
            out_file.write(fmt_.format(*paras))

            # Write out information about the optimization parameters directly.
            out_file.write('\n Optimization Parameters\n\n')
            fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
            out_file.write(fmt_.format(*['Identifier', 'Start', 'Step', 'Current']))
            fmt_ = '{0:>15}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
            for i, _ in enumerate(paras_curre):
                paras = [i, paras_start[i], paras_steps[i], paras_curre[i]]
                out_file.write(fmt_.format(*paras))

            # Write out the current covariance matrix of the reward shocks.
            out_file.write('\n\n Covariance Matrix \n\n')

            for which in ['Start', 'Step', 'Current']:
                if which == 'Start':
                    paras = paras_start
                elif which == 'Step':
                    paras = paras_steps
                else:
                    paras = paras_curre
                fmt_ = '{0:>15}   \n\n'
                out_file.write(fmt_.format(*[which]))
                shocks_cov = dist_optim_paras(paras, True)[4]
                fmt_ = '{0:15.4f}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
                for i in range(4):
                    out_file.write(fmt_.format(*shocks_cov[i, :]))
                out_file.write('\n')

            fmt_ = '\n{0:<25}{1:>15}\n'
            out_file.write(fmt_.format(*[' Number of Steps', num_steps]))
            out_file.write(fmt_.format(*[' Number of Evaluations', num_evals]))

    def _logging_final(self, rslt):
        """ This method writes out some information when the optimization is
        finished.
        """
        fmt_ = '{0:<10} {1:<25}\n'
        with open('optimization.robupy.log', 'a') as out_file:
            out_file.write('Final Report\n\n')
            out_file.write(fmt_.format('Success', str(rslt['success'])))
            out_file.write(fmt_.format('Message', rslt['message']))
            out_file.write(fmt_.format('Criterion', self.attr['value_steps']))

        with open('optimization.robupy.info', 'a') as out_file:
            out_file.write('\n TERMINATED')


