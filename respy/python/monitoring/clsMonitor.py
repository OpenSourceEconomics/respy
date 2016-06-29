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
    """ This class monitors the progress of the estimation and feeds it into
    the main program.
    """
    def __init__(self):

        self.proc = None

    def start(self):
        """ This method starts the daemon process after some initial cleanup.
        """

        if os.path.exists('est.respy.log'):
            os.unlink('est.respy.log')

        if os.path.exists('est.respy.info'):
            os.unlink('est.respy.info')

        if os.path.exists('est.respy.paras'):
            os.unlink('est.respy.paras')

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
        args = get_information()
        update_information(*args)

        with open('est.respy.info', 'a') as out_file:
            out_file.write('\n TERMINATED')

        # Return the final values
        fval, x = args[4], args[5]

        return x, fval

    def _terminate(self):
        """ Terminate daemon process.
        """
        try:
            self.proc.terminate()
            self.proc.communicate()
        except OSError:
            pass
