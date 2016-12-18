#!/usr/bin/env python
""" This script allows to compare the performance of alternative decision rules.
"""

from auxiliary_performance import plot
from auxiliary_performance import run

''' Execution of module as script.
'''

if __name__ == '__main__':

    run([0.00, 0.05])

    plot()


