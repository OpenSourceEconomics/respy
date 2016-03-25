""" The tests in this module compare the ROBUPY package to the original
RESTUD code for the special cases where they overlap.
"""

# standard library
from pandas.util.testing import assert_frame_equal

import pandas as pd
import numpy as np

import pytest
import shutil
import os

# ROBUPY import
from robupy.tests.codes.random_init import generate_random_dict
from robupy.tests.codes.random_init import print_random_dict

from robupy.shared.auxiliary import distribute_class_attributes

from robupy import simulate
from robupy import solve
from robupy import read


''' Auxiliary functions
'''


def transform_robupy_to_restud(model_paras, level, edu_start, edu_max,
        num_agents, num_periods, num_draws_emax, delta):
    """ Transform a ROBUPY initialization file to a RESTUD file.
    """
    # Ensure restrictions
    assert (level == 0.00)
    assert (edu_start == 10)
    assert (edu_max == 20)

    # Write to initialization file
    with open('in.txt', 'w') as file_:

        # Write out some basic information about the problem.
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
        coeffs = edu_coeffs + [home, delta]
        line = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f}' \
                ' {4:10.6f}\n'.format(*coeffs)
        file_.write(line)

        # Write out coefficients of correlation, which need to be constructed
        # based on the covariance matrix.
        shocks_cov = model_paras['shocks_cov']
        rho = np.identity(4)
        rho_10 = shocks_cov[1][0] / (np.sqrt(shocks_cov[1][1]) * np.sqrt(shocks_cov[0][0]))
        rho_20 = shocks_cov[2][0] / (np.sqrt(shocks_cov[2][2]) * np.sqrt(shocks_cov[0][0]))
        rho_30 = shocks_cov[3][0] / (np.sqrt(shocks_cov[3][3]) * np.sqrt(shocks_cov[0][0]))
        rho_21 = shocks_cov[2][1] / (np.sqrt(shocks_cov[2][2]) * np.sqrt(shocks_cov[1][1]))
        rho_31 = shocks_cov[3][1] / (np.sqrt(shocks_cov[3][3]) * np.sqrt(shocks_cov[1][1]))
        rho_32 = shocks_cov[3][2] / (np.sqrt(shocks_cov[3][3]) * np.sqrt(shocks_cov[2][2]))
        rho[1, 0] = rho_10; rho[2, 0] = rho_20; rho[3, 0] = rho_30
        rho[2, 1] = rho_21; rho[3, 1] = rho_31; rho[3, 2] = rho_32
        for j in range(4):
            line = ' {0:10.5f} {1:10.5f} {2:10.5f} ' \
                   ' {3:10.5f}\n'.format(*rho[j, :])
            file_.write(line)

        # Write out standard deviations. Scaling for standard deviations in the
        # education and home equation required. This is undone again in the
        # original FORTRAN code.
        sigmas = np.sqrt(np.diag(shocks_cov)); sigmas[2:] = sigmas[2:]/1000
        line = '{0:10.5f} {1:10.5f} {2:10.5f} {3:10.5f}\n'.format(*sigmas)
        file_.write(line)


''' Main
'''


@pytest.mark.usefixtures('fresh_directory', 'set_seed', 'supply_resources')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """  Compare results from the RESTUD program and the ROBUPY package.
        """
        # Prepare RESTUD program
        tmp_dir = os.getcwd()

        file_dir = os.path.dirname(os.path.realpath(__file__))

        os.chdir(file_dir + '/codes')

        # Create required directory structure.
        if not os.path.exists('build'):
            os.mkdir('build')

        if not os.path.exists('../lib/'):
            os.mkdir('../lib/')

        # Build the upgraded version of the original Keane & Wolpin (1994)
        # codes.
        os.chdir('build')

        shutil.copy('../dp3asim.f95', '.')

        os.system(' gfortran -fcheck=bounds -o dp3asim dp3asim.f95 >'
                  ' /dev/null 2>&1')

        shutil.copy('dp3asim', '../../lib/')

        os.chdir('../')

        shutil.rmtree('build')

        # Return to the temporary directory.
        os.chdir(tmp_dir)

        # Impose some constraints on the initialization file which ensures that
        # the problem can be solved by the RESTUD code. The code is adjusted to
        # run with zero draws.
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
        model_paras, level, edu_start, edu_max, num_agents, num_periods, \
            num_draws_emax, delta = \
                distribute_class_attributes(robupy_obj,
                    'model_paras', 'level', 'edu_start', 'edu_max', 'num_agents',
                    'num_periods', 'num_draws_emax', 'delta')

        transform_robupy_to_restud(model_paras, level, edu_start, edu_max,
            num_agents, num_periods, num_draws_emax, delta)

        # Solve model using RESTUD code.
        os.system(file_dir + '/lib/dp3asim > /dev/null 2>&1')

        # Solve model using ROBUPY package.
        solve(robupy_obj)
        simulate(robupy_obj)

        # Compare the simulated datasets generated by the programs.
        py = pd.DataFrame(np.array(np.genfromtxt('data.robupy.dat',
                missing_values='.'), ndmin=2)[:, -4:])

        fort = pd.DataFrame(np.array(np.genfromtxt('ftest.txt',
                missing_values='.'), ndmin=2)[:, -4:])

        assert_frame_equal(py, fort)
