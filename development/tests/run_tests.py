#!/usr/bin/env python
""" This module is used for the development setup.
"""

# project library
import sys
import os

# module-wide variables
HOME = os.getcwd()

''' Run fixed test battery
'''
os.chdir('fixed')

os.system('python run')

os.chdir(HOME)

''' Run randon test battery
'''
os.chdir('random')

os.system('./run --hours 0.001')

os.chdir(HOME)




