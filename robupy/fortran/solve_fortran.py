""" This module contains all the capabilities to solve the dynamic
programming problem.
"""

# standard library
import os

# module-wide variables
PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))

''' Public function
'''


def solve_fortran(robupy_obj):
    """ Solve dynamic programming using FORTRAN.
    """
    # Distribute class attributes
    init_dict = robupy_obj.get_attr('init_dict')

    write_robufort_initialization(init_dict)

    os.system('"' + PACKAGE_PATH + '/bin/robufort"')

    # Finishing
    return  None


''' Auxiliary function
'''


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
        line = ' {0:15.10f}\n'.format(init_dict['HOME']['int'])
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

