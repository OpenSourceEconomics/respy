""" This module manages the logging of the estimation progress across
implementations and optimizers.
"""


import subprocess
import sys
import os

import atexit

from respy.python.shared.shared_constants import ROOT_DIR

from respy.python.monitoring.monitoring_child import update_information
from respy.python.monitoring.monitoring_child import get_information

class MonitorCls(object):


    def __init__(self):
        # Distribute class attributes
        self.attr = dict()

        self.attr['working'] = True

        self.proc = None

    def start(self):
        """
        """

        if os.path.exists('optimization.respy.log'):
            os.unlink('optimization.respy.log')

        if os.path.exists('optimization.respy.info'):
            os.unlink('optimization.respy.info')

        fname = ROOT_DIR + '/python/monitoring/monitoring_child.py'
        self.proc = subprocess.Popen([sys.executable, fname])
        atexit.register(self._terminate)

    def _terminate(self):

        self.proc.terminate()
        self.proc.communicate()

    def stop(self):

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

# TODO: Incorporate?
#        fmt_ = '{0:<10} {1:<25}\n'
#        with open('optimization.respy.log', 'a') as out_file:
#            out_file.write('Final Report\n\n')
#            out_file.write(fmt_.format('Success', True))
#            out_file.write(fmt_.format('Message', 'update messages'))
#            out_file.write(fmt_.format('Criterion', self.attr['value_steps']))
#            out_file.write(fmt_.format('Time', time.ctime()))

            #     np.savetxt(open('paras_steps.respy.log', 'wb'), x,
            # fmt='%15.8f')
            #     with open('optimization.respy.log', 'a') as out_file:
            #         fmt_ = '{0:<10} {1:<25}\n'
            #         out_file.write(fmt_.format('Step', int(num_steps) + 1))
            #         out_file.write(fmt_.format('Criterion', crit_val))
            #         out_file.write(fmt_.format('Time', time.ctime()))
            #         out_file.write('\n\n')

            # Update class attributes