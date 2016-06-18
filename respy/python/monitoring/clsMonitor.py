""" This module manages the logging of the estimation progress across
implementations and optimizers.
"""

# standard library
import subprocess
import atexit
import sys
import os

# project library
from respy.python.monitoring.monitoring_child import update_information
from respy.python.monitoring.monitoring_child import get_information
from respy.python.shared.shared_constants import ROOT_DIR


class MonitorCls(object):

    def __init__(self):

        self.proc = None

    def start(self):
        """ This method starts the daemon process after some initial cleanup.
        """

        if os.path.exists('optimization.respy.log'):
            os.unlink('optimization.respy.log')

        if os.path.exists('optimization.respy.info'):
            os.unlink('optimization.respy.info')

        for which in ['start', 'step', 'current']:
            fname = 'opt_info_' + which + '.respy.log'
            if os.path.exists(fname):
                os.unlink(fname)

        # Start a subprocess and make sure that it is terminated even in the
        # event of an unexpected shutdown of the main process. This makes
        # ensures that there are no zombie processes.
        fname = ROOT_DIR + '/python/monitoring/monitoring_child.py'
        self.proc = subprocess.Popen([sys.executable, fname])
        atexit.register(self._terminate)

    def stop(self):
        """ Stop the daemon process and make sure that the information
        provided to the main process is updated with the most recent results.
        """

        self._terminate()

        # We make sure that the final information is up to date.
        args = []
        for which in ['start', 'step', 'current']:
            args += get_information(which)
        update_information(*args)

        with open('optimization.respy.info', 'a') as out_file:
            out_file.write('\n TERMINATED')

        # Return the final values
        fval, x = get_information('step')[1:]

        return x, fval

    def _terminate(self):
        """ Terminate daemon process.
        """
        try:
            self.proc.terminate()
            self.proc.communicate()
        except OSError:
            pass
