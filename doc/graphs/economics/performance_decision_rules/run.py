#!/usr/bin/env python
""" This script allows to compare the performance of alternative decision rules.
"""

from auxiliary_performance import plot
from auxiliary_performance import run

''' Execution of module as script.
'''

if __name__ == '__main__':

    # We only analyze a subset of alternative decision rules.
    run([0.00, 0.015])

    import pickle as pkl

    rslt = pkl.load(open('performance.respy.pkl', 'rb'))
    print(rslt)
    #plot()


