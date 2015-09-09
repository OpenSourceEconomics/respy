#!/usr/bin/env python

""" This module is used for the development setup.
"""
# standard library
from pandas.util.testing import assert_frame_equal
import pandas as pd
import numpy as np
import sys
import os
import shutil

# project libra
# project library
import time
import sys
import os

sys.path.insert(0, os.environ['ROBUPY'])

sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')


# project library
from robupy import read, simulate
from robupy.python.solve_python import solve_python
from robupy.fortran.solve_fortran import solve_fortran
from robupy.tests.random_init import generate_random_dict, print_random_dict

# Relative Criterion
HOME = os.getcwd()

CURRENT_WD = os.environ['ROBUPY'] + '/development'
ROBUPY_DIR = os.environ['ROBUPY'] + '/robupy'


from modules.auxiliary import transform_robupy_to_restud


for _ in range(1):

    # Re-compile ROBUPY package
    #os.chdir('/home/peisenha/robustToolbox/package/robupy')

    #os.system('./waf distclean')

    #os.system('./waf configure build --fast')

    #os.chdir(HOME)

    # Run
#    os.system('robupy-solve --simulate')

    while True:

        constraints = dict()
        # THIS shoudl work with PYTHON as well?
        constraints['version'] = np.random.choice(['F2PY'])
        constraints['level'] = 0.00
        constraints['eps_level'] = True
        init_dict = generate_random_dict(constraints)

        num_agents = init_dict['SIMULATION']['agents']
        num_draws = init_dict['SOLUTION']['draws']
        if num_draws < num_agents:
            init_dict['SOLUTION']['draws'] = num_agents

        print_random_dict(init_dict)

        os.system('gfortran -O3 -o dp3asim dp3asim.f95')





        robupy_obj = read('model.robupy.ini')

        init_dict = robupy_obj.get_attr('init_dict')

        transform_robupy_to_restud(init_dict)



        start_time = time.time()

        os.system('./dp3asim')

        print('RESTUD: ', time.time() - start_time)


        # ROBUFORT
        os.chdir(ROBUPY_DIR)

        os.system('./waf distclean; ./waf configure build --fast')

        os.chdir(CURRENT_WD)


        start_time = time.time()

        solve_fortran(robupy_obj)

        print('ROBUFORT: ', time.time() - start_time)

        break