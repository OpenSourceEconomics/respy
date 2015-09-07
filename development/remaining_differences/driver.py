
""" This modules contains some additional tests that are only used in
long-run development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal
import sys
import os

# project library
import time

# Development Imports


# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
sys.path.insert(0, os.environ['ROBUPY'] + '/development/tests/random')
from modules.auxiliary import compile_package, transform_robupy_to_restud



from robupy import read, solve, simulate
from robupy.tests.random_init import generate_init

print('not compiling at the moment')
#compile_package('fast')


for _ in range(1):
    constr = dict()
    constr['fast'] = 'False'
    constr['level'] = 0.00
    constr['edu'] = (10, 20)

    #generate_init(constr)


    robupy_obj = read('model.robupy.ini')

    init_dict = robupy_obj.get_attr('init_dict')


    with open('model.robufort.ini', 'w') as file_:

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





    filenames = ['robupy_program_constants.f90', 'robupy_auxiliary.f90',
                     'robupy_core.f90','robufort.f90']
    #with open('robufort_extended.f90', 'w') as outfile:
    #        for fname in filenames:
    #              with open(fname) as infile:
    #                for line in infile:
    #                    outfile.write(line)
    #            outfile.write('\n')

    import shlex
    # Read in all subroutines
    subroutines = dict()

    for f90 in ['robupy_core.f90', 'robupy_auxiliary.f90']:

        with open('robupy_core.f90') as file_:

            for _ in range(2000):

                list_ = shlex.split(file_.readline())

                # Initialize new subroutine
                new_subroutine = False
                try:
                    new_subroutine = (list_[0] == 'SUBROUTINE')
                except IndexError:
                    pass

                if new_subroutine:
                    name = list_[1].split('(')[0]
                    subroutines[name] = []

                # Collect algorithm
                is_algorithm = ('Algorithm' in list_)

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


    for line in subroutines['backward_induction']:
        print(line)


    old_file = open('robufort.f90', 'r')
    new_file = open('robufort_inlining.f90', 'w')

    for _ in range(2000):

        old_line = old_file.readline()

        old_list = shlex.split(old_line)


        try:
            is_backward = (old_list[0] == 'CALL') and  \
            (old_list[1].split('(')[0] == 'backward_induction')

        except IndexError:
            is_backward = False


        if is_backward:
            new_file.write('INLINING: backward_induction\n')

            while True:

                old_list = shlex.split(old_file.readline())

                is_end = ')' in old_list[-1]
                print( old_list[-1])
                if is_end:
                    break

        else:
            print(old_line)
            new_file.write(old_line)

    old_file.close()
    new_file.close()



    # Replacements
    old_file = open('robufort_inlining.f90', 'r')
    new_file = open('robufort_extended.f90', 'w')

    for _ in range(2000):

        line_ = old_file.readline()



        is_inlining = 'INLINING' in line_


        if is_inlining:

            for code_line in subroutines['backward_induction']:
                new_file.write(code_line)

        else:

            new_file.write(line_)

    new_file.close(); new_file.close()

#    print(subroutines['backward_induction'])
    #print(subroutines['backward_induction'])
#    os.system('gfortran -finline-limit=800 -O3 -o robufort '
#              'robufort_extended.f90')
    try:
        os.remove('robufort')
    except:
        pass

    os.system('gfortran -finline-limit=800 -O3 -o robufort '
              'robufort_extended.f90')

    transform_robupy_to_restud(init_dict)

    os.system(' gfortran -O3 -o dp3asim dp3asim.f90')

    # Run DP3ASIM

    start_time = time.time()

    os.system('./dp3asim')

    print('RESTUD ', time.time() - start_time)



    start_time = time.time()

    os.system('./robufort')

    print('ROBUPY ', time.time() - start_time)

