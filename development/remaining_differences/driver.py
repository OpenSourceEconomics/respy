
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







def mark_inlinings(subroutines):
    """ This function marks the subroutines for which inlining
    information is available.
    """
    # Auxiliary objects
    inlining_routines = subroutines.keys()

    # Read file with pre-inlining code
    with open('robufort_extended.f90', 'r') as old_file:
        num_lines = len(old_file.readlines())

    # Initialize logical variables
    is_program = False

    with open('robufort_extended.f90', 'r') as old_file:
        with open('.robufort_inlining.f90', 'w') as new_file:

            for _ in range(num_lines):

                # Extract old information
                old_line = old_file.readline()
                old_list = shlex.split(old_line)

                # Skip all empty lines
                if not old_list:
                    new_file.write(old_line)
                    continue

                # Skip modifying all lines before actual program begins.
                # This skips over the module where the program constants
                # are defined.
                if not is_program:
                    is_program = (old_list[0] == 'PROGRAM')
                    new_file.write(old_line)
                    continue

                # Skip modifying of all lines without a CALL statement
                is_call = (old_list[0] == 'CALL')
                if not is_call:
                    new_file.write(old_line)
                    continue

                # Determine name is call that will be replaced. Note that
                # not all functions or routines will be relaced.
                name = old_list[1].split('(')[0]
                if name not in inlining_routines:
                    new_file.write(old_line)
                    continue

                # Write out keyworkd for future replacement. Ensure that
                # interfaces that run across multiple lines are removed
                # completely.
                new_file.write('INLINING: ' + name + ' \n')
                while True:
                    is_end = ')' in old_list[-1]
                    if is_end:
                        break
                    old_list = shlex.split(old_file.readline())


def replace_inlinings():
    """ This function replaces subroutines marked for inlining with the
    relevant code.
    """
    # Auxiliary objects
    count = 0

    # Read file with inlining instructions
    with open('.robufort_inlining.f90', 'r') as old_file:
        old_lines = old_file.readlines()

    # Construct new FORTRAN file.
    with open('robufort_extended.f90', 'w') as new_file:
        for old_line in old_lines:

            # Check for subroutines marked for replacement
            is_inlining = 'INLINING' in old_line

            if not is_inlining:
                new_file.write(old_line)
            else:
                # Write out code of relevant subroutine
                name = shlex.split(old_line)[1]
                for code_line in subroutines[name]:
                    new_file.write(code_line)
                # Store workload
                count += 1

    # Finishing
    return count


def write_robufort_initialization(init_dict):
    """ Write out model request to hidden file .model.robufort.ini.
    """

    with open('.model.robufort.ini', 'w') as file_:

        # BASICS
        line = '{0:10d}\n'.format(init_dict['BASICS']['periods'])
        file_.write(line)

        line = '{0:15.10f}\n'.format(init_dict['BASICS']['delta'])
        file_.write(line)

        # WORK
        for label in ['A', 'B']:
            num = [init_dict[label]['int']] + init_dict[label]['coeff']
            line = ' {0:15.10f} {1:15.10f} {2:15.10f} {3:15.10f}  {4:15.10f}' \
                        ' {5:15.10f}\n'.format(*num)
            file_.write(line)

        # EDUCATION
        num = [init_dict['EDUCATION']['int']] + init_dict['EDUCATION']['coeff']
        line = ' {0:15.10f} {1:15.10f} {2:15.10f}\n'.format(*num)
        file_.write(line)

        line = '{0:10d} '.format(init_dict['EDUCATION']['start'])
        file_.write(line)

        line = '{0:10d}\n'.format(init_dict['EDUCATION']['max'])
        file_.write(line)

        # HOME
        line = '{0:15.10f}\n'.format(init_dict['HOME']['int'])
        file_.write(line)

        # SHOCKS
        shocks = init_dict['SHOCKS']
        for j in range(4):
            line = ' {0:15.5f} {1:15.5f} {2:15.5f} {3:15.5f}\n'.format(*shocks[j])
            file_.write(line)

         # SOLUTION
        line = '{0:10d}\n'.format(init_dict['SOLUTION']['draws'])
        file_.write(line)

        line = '{0:10d}\n'.format(init_dict['SOLUTION']['seed'])
        file_.write(line)

        # SIMULATION
        line = '{0:10d}\n'.format(init_dict['SIMULATION']['agents'])
        file_.write(line)

        line = '{0:10d}\n'.format(init_dict['SIMULATION']['seed'])
        file_.write(line)


def read_subroutines():
    """ Read information on all subroutines which are candidates for
    inlining.
    """
    # Initialize container
    subroutines = dict()

    # Determine number of lines
    with open('robupy_core.f90', 'r') as old_file:
        num_lines = len(old_file.readlines())

    # Extract information
    with open('robupy_core.f90') as file_:

        for _ in range(num_lines):

            list_ = shlex.split(file_.readline())

            # Skip all empty lines
            if not list_:
                continue

            # Initialize container for new subroutine
            new_subroutine = (list_[0] == 'SUBROUTINE')
            if new_subroutine:
                name = list_[1].split('(')[0]
                subroutines[name] = []

            # Collect algorithm information.
            is_algorithm = ('Algorithm' in list_)

            # The WHILE loop iterates over all lines of the file until
            # the subroutine ends.
            if is_algorithm:
                while True:
                    code_line = file_.readline()
                    list_ = shlex.split(code_line)

                    is_end = False
                    try:
                        is_end = list_[:2] == ['END', 'SUBROUTINE']
                    except IndexError:
                        pass

                    if is_end:
                        break

                    # Collect information
                    subroutines[name] += [code_line]

    # Finishing
    return subroutines


print('not compiling at the moment')
#compile_package('fast')


''' Compilation of ROBUFORT
'''
try:
        os.remove('robufort')
except:
        pass

# Performance considerations require an automatic inlining of the core
# subroutines in a single file. Prepare starting version of extended ROBUFORT
# code. At first the module containing the program constants is inserted.
# Then the code tailored for the inlinings is added.
with open('robufort_extended.f90', 'w') as outfile:
    for fname in ['robupy_program_constants.f90', 'robufort.f90']:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
        outfile.write('\n')

# This loop iteratively marks subroutines for inlinings and then replaces
# them with the relevant code lines. The loop stops once no subroutines
# are marked for further inlining.
subroutines = read_subroutines()

while True:

    mark_inlinings(subroutines)

    count = replace_inlinings()

    # Check for further applicability and cleaning.
    if count == 0:
        os.remove('.robufort_inlining.f90')
        break

# Compile the executable
os.system('gfortran -O3 -o robufort robufort_extended.f90')


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

