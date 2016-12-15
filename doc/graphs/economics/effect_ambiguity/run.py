#!/usr/bin/env python
""" This script plots how choices change as we increase the level of ambiguity faced by agents.
"""

from auxiliary_choices import plot
from auxiliary_choices import run

''' Execution of module as script.
'''

if __name__ == '__main__':

    run()

    # We only plot a subset of points from the grid.
    plot([0.0, 0.004, 0.02])
