""" The tests in this module compare the ROBUPY package to the original
RESTUD code for the special cases where they overlap.
"""

# standard library
from pandas.util.testing import assert_frame_equal

import pandas as pd
import numpy as np

import pytest
import sys
import os

# testing library
from material.auxiliary import distribute_model_description
from material.auxiliary import compile_package

# ROBUPY import
ROBUPY_DIR = os.environ['ROBUPY']
sys.path.insert(0, ROBUPY_DIR)

from robupy.tests.random_init import generate_random_dict
from robupy.tests.random_init import print_random_dict

from robupy import *


''' Auxiliary functions
'''


def transform_robupy_to_restud(model_paras, init_dict):
    """ Transform a ROBUPY initialization file to a RESTUD file.
    """
    # Ensure restrictions
    assert (init_dict['AMBIGUITY']['level'] == 0.00)
    assert (init_dict['EDUCATION']['start'] == 10)
    assert (init_dict['EDUCATION']['max'] == 20)

    # Write to initialization file
    with open('in.txt', 'w') as file_:

        # Write out some basic information about the problem.
        num_agents = init_dict['SIMULATION']['agents']
        num_periods = init_dict['BASICS']['periods']
        num_draws_emax = init_dict['SOLUTION']['draws']
        file_.write(' {0:03d} {1:05d} {2:06d} {3:06f}'
            ' {4:06f}\n'.format(num_periods, num_agents, num_draws_emax,-99.0,
            500.0))

        # Write out coefficients for the two occupations.
        coeffs_a, coeffs_b = model_paras['coeffs_a'], model_paras['coeffs_b']
        for coeffs in [coeffs_a, coeffs_b]:
            line = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f}  {4:10.6f}' \
                    ' {5:10.6f}\n'.format(*coeffs)
            file_.write(line)

        # Write out coefficients for education and home payoffs as well as
        # the discount factor. The intercept is scaled. This is later undone
        # again in the original FORTRAN code.
        coeffs_edu = model_paras['coeffs_edu']
        coeffs_home = model_paras['coeffs_home']

        edu_int = coeffs_edu[0] / 1000; edu_coeffs = [edu_int]
        home = coeffs_home[0] / 1000
        for j in range(2):
            edu_coeffs += [-coeffs_edu[j + 1] / 1000]
        delta = init_dict['BASICS']['delta']
        coeffs = edu_coeffs + [home, delta]
        line = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f}' \
                ' {4:10.6f}\n'.format(*coeffs)
        file_.write(line)

        # Write out coefficients of correlation, which need to be constructed
        # based on the covariance matrix.
        shocks = model_paras['shocks']; rho = np.identity(4)
        rho_10 = shocks[1][0] / (np.sqrt(shocks[1][1]) * np.sqrt(shocks[0][0]))
        rho_20 = shocks[2][0] / (np.sqrt(shocks[2][2]) * np.sqrt(shocks[0][0]))
        rho_30 = shocks[3][0] / (np.sqrt(shocks[3][3]) * np.sqrt(shocks[0][0]))
        rho_21 = shocks[2][1] / (np.sqrt(shocks[2][2]) * np.sqrt(shocks[1][1]))
        rho_31 = shocks[3][1] / (np.sqrt(shocks[3][3]) * np.sqrt(shocks[1][1]))
        rho_32 = shocks[3][2] / (np.sqrt(shocks[3][3]) * np.sqrt(shocks[2][2]))
        rho[1, 0] = rho_10; rho[2, 0] = rho_20; rho[3, 0] = rho_30
        rho[2, 1] = rho_21; rho[3, 1] = rho_31; rho[3, 2] = rho_32
        for j in range(4):
            line = ' {0:10.5f} {1:10.5f} {2:10.5f} ' \
                   ' {3:10.5f}\n'.format(*rho[j, :])
            file_.write(line)

        # Write out standard deviations. Scaling for standard deviations in the
        # education and home equation required. This is undone again in the
        # original FORTRAN code.
        sigmas = np.sqrt(np.diag(shocks)); sigmas[2:] = sigmas[2:]/1000
        line = '{0:10.5f} {1:10.5f} {2:10.5f} {3:10.5f}\n'.format(*sigmas)
        file_.write(line)


''' Main
'''


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass:
    def test_1(self):
        """  Compare results from the RESTUD program and the ROBUPY package.
        """

        # Ensure that fast solution methods are available
        compile_package('--fortran --debug', True)

        # Prepare RESTUD program
        os.chdir(ROBUPY_DIR + '/material')
        os.system(' gfortran -fcheck=bounds -o dp3asim dp3asim.f95 >'
                  ' /dev/null 2>&1')
        os.remove('pei_additions.mod')
        os.remove('imsl_replacements.mod')
        os.chdir('../')

        # Impose some constraints on the initialization file which ensures that
        # the problem can be solved by the RESTUD code. The code is adjusted to
        # run with zero disturbances.
        constraints = dict()
        constraints['edu'] = (10, 20)
        constraints['level'] = 0.00
        constraints['is_deterministic'] = True

        # Generate random initialization file. The RESTUD code uses the same
        # random draws for the solution and simulation of the model. Thus,
        # the number of draws is required to be less or equal to the number
        # of agents.
        init_dict = generate_random_dict(constraints)

        num_agents = init_dict['SIMULATION']['agents']
        num_draws_emax = init_dict['SOLUTION']['draws']
        if num_draws_emax < num_agents:
            init_dict['SOLUTION']['draws'] = num_agents

        print_random_dict(init_dict)

        # Indicate RESTUD code the special case of zero disturbance.
        open('.restud.testing.scratch', 'a').close()

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        # This flag aligns the random components between the RESTUD program and
        # ROBUPY package. The existence of the file leads to the RESTUD program
        # to write out the random components.
        model_paras, init_dict = distribute_model_description(robupy_obj,
                    'model_paras', 'init_dict')

        transform_robupy_to_restud(model_paras, init_dict)

        # Solve model using RESTUD code.
        os.system('./' + ROBUPY_DIR + '/material/dp3asim > /dev/null 2>&1')

        # Solve model using ROBUPY package.
        solve(robupy_obj)

        # Compare the simulated datasets generated by the programs.
        py = pd.DataFrame(np.array(np.genfromtxt('data.robupy.dat',
                missing_values='.'), ndmin=2)[:, -4:])

        fort = pd.DataFrame(np.array(np.genfromtxt('ftest.txt',
                missing_values='.'), ndmin=2)[:, -4:])

        assert_frame_equal(py, fort)
