#!/usr/bin/env python
""" This script allows to test the original codes against the ROBUPY package.
"""

# standard library
import numpy as np
import sys
import os

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.tests.random_init import generate_init

from robupy import read, solve, simulate


""" Compile toolbox
"""
#os.system(' gfortran -o dp3asim dp3asim.f95')

#os.system('./dp3asim')

for _ in range(10):

    # Generate constraints
    constraints = dict()
    constraints['level'] = 0.00

    # Generate random initialization file
    generate_init(constraints)

    # Perform toolbox actions
    robupy_obj = read('model.robupy.ini')


    init_dict = robupy_obj.get_attr('init_dict')

    num_periods = init_dict['BASICS']['periods']
    num_agents = init_dict['SIMULATION']['agents']
    num_draws = init_dict['SOLUTION']['draws']
    tau = 0.4

    delta = init_dict['BASICS']['delta']

    coeffs_a = [init_dict['A']['int']] + init_dict['A']['coeff']
    coeffs_b = [init_dict['B']['int']] + init_dict['B']['coeff']

    # Home
    home = init_dict['HOME']['int'] / 1000

    # Education
    edu_int = init_dict['EDUCATION']['int'] / 1000
    print(init_dict['EDUCATION']['coeff'])

    edu_coeffs = [edu_int]
    for i in range(2):
        edu_coeffs += [init_dict['EDUCATION']['coeff'][i] / 1000]


    f = open('in1.txt', 'w')

    f.write("{0:03d} {1:05d} {2:06d} {3:06d}  {4:06d}\n".format(num_periods,
                num_agents, num_draws, -99, -99))

    line = '{0} {1} {2} {3}  {4} {5}\n'.format(*coeffs_a)
    f.write(line)

    line = '{0} {1} {2} {3}  {4} {5}\n'.format(*coeffs_b)
    f.write(line)

    input = edu_coeffs + [home, delta]
    line = '{0} {1} {2} {3}  {4}\n'.format(*input)
    f.write(line)


    shocks = init_dict['SHOCKS']

    covs = np.identity(4)
    rho_01 = shocks[0][1] / (np.sqrt(shocks[0][0]) * np.sqrt(shocks[1][1]))
    rho_12 = shocks[0][1] / (np.sqrt(shocks[0][0]) * np.sqrt(shocks[1][1]))
    rho_01 = shocks[0][1] / (np.sqrt(shocks[0][0]) * np.sqrt(shocks[1][1]))

#    print(covs)

    f.close()

    print(line)