
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

CURRENT_WD = os.environ['ROBUPY'] + '/development/remaining_differences'
ROBUPY_DIR = os.environ['ROBUPY'] + '/robupy'

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
from modules.auxiliary import compile_package, transform_robupy_to_restud


from robupy import read, solve, simulate
from robupy.tests.random_init import generate_init

from robupy.fortran.solve_fortran import write_robufort_initialization

for _ in range(1):

    compile_package("fast")

    constr = dict()
    constr['fast'] = 'False'
    constr['level'] = 0.00
    constr['edu'] = (10, 20)

    #generate_init(constr)


    robupy_obj = read('model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')



    # Run DP3ASIM
    transform_robupy_to_restud(init_dict)

    os.system(' gfortran -O3 -o dp3asim dp3asim.f90')

    start_time = time.time()

    os.system('./dp3asim')

    print('RESTUD ', time.time() - start_time)


    # Run ROBUFORT
    os.chdir(ROBUPY_DIR)

    os.system('./waf distclean; ./waf configure build --fast')

    os.chdir(CURRENT_WD)

    write_robufort_initialization(init_dict)

    start_time = time.time()

    os.system('./robufort')

    print('ROBUPY ', time.time() - start_time)

