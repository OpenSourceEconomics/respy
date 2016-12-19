#!/usr/bin/env python
""" This script plots how choices change as we increase the level of ambiguity
faced by agents.
"""

from auxiliary_effects import plot
from auxiliary_effects import run

''' Execution of module as script.
'''

if __name__ == '__main__':

    run()

    plot([0.0, 0.01, 0.02])
