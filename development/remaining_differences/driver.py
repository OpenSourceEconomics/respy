
""" This modules contains some additional tests that are only used in
long-run development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal
import fileinput
import shutil
import sys
import os

# project library
import time
import shlex
# Development Imports


# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
from modules.auxiliary import compile_package, transform_robupy_to_restud



from robupy import read, solve, simulate
from robupy.tests.random_init import generate_init


for _ in range(1):
    constr = dict()
    constr['fast'] = 'False'
    constr['level'] = 0.00
    constr['edu'] = (10, 20)

    generate_init(constr)


    robupy_obj = read('model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')




    transform_robupy_to_restud(init_dict)

    os.system(' gfortran -O3 -o dp3asim dp3asim.f90')

    # Run DP3ASIM

    start_time = time.time()

    os.system('./dp3asim')

    print('RESTUD ', time.time() - start_time)


    # Run ROBUFORT
    write_robufort_initialization(init_dict)

    start_time = time.time()

    os.system('./robufort')

    print('ROBUPY ', time.time() - start_time)

