#!/usr/bin/env python
""" This module starts the test runs using the data specifications from
the seminal paper ``The Solution and Estimation of Discrete Choice Dynamic
Programming Models by Simulation and Interpolation: Monte Carlo Evidence''
by Michael Keane and Kenneth Wolpin in The Review of Economics and Statistics.
"""

# standard library
import os


for dir_ in ['one', 'two', 'three']:

    os.chdir('data_' + dir_)

    os.system('robupy-clean &')

    os.system('robupy-solve --simulate &')

    os.chdir('..')