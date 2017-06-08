from pandas.util.testing import assert_frame_equal
import pandas as pd
import numpy as np
import subprocess
import pytest

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.shared.shared_constants import IS_FORTRAN
from codes.random_init import generate_random_dict
from codes.auxiliary import simulate_observed
from respy import RespyCls


def transform_respy_to_restud(optim_paras, edu_spec, num_agents_sim, num_periods,
        num_draws_emax):
    """ Transform a RESPY initialization file to a RESTUD file.
    """
    # Ensure restrictions
    assert (edu_spec['start'][0] == 10)
    assert (edu_spec['max'] == 20)

    # Write to initialization file
    with open('in.txt', 'w') as file_:

        # Write out some basic information about the problem.
        file_.write(' {0:03d} {1:05d} {2:06d} {3:06f} {4:06f}\n'.format(
            num_periods, num_agents_sim, num_draws_emax, -99.0, 500.0))

        # Write out coefficients for the two occupations.
        coeffs_a, coeffs_b = optim_paras['coeffs_a'], optim_paras['coeffs_b']
        for coeffs in [coeffs_a, coeffs_b]:
            line = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f} {4:10.6f} {5:10.6f}\n'.format(*coeffs)
            file_.write(line)

        # Write out coefficients for education and home rewards as well as the discount factor.
        # The intercept is scaled. This is later undone again in the original FORTRAN code.
        coeffs_edu = optim_paras['coeffs_edu']
        coeffs_home = optim_paras['coeffs_home']

        edu_int = coeffs_edu[0] / 1000
        edu_coeffs = [edu_int]
        home = coeffs_home[0] / 1000
        for j in range(2):
            edu_coeffs += [-coeffs_edu[j + 1] / 1000]
        coeffs = edu_coeffs + [home, optim_paras['delta'][0]]
        fmt = ' {0:10.6f} {1:10.6f} {2:10.6f} {3:10.6f} {4:10.6f}\n'
        line = fmt.format(*coeffs)
        file_.write(line)

        # Write out coefficients of correlation and standard deviations in the standard deviations
        # in the education and home equation required. This is undone again in the original
        # FORTRAN code. All this is working only under the imposed absence of any randomness.
        rho = np.zeros((4, 4))
        for j in range(4):
            line = ' {0:10.5f} {1:10.5f} {2:10.5f} ' \
                   ' {3:10.5f}\n'.format(*rho[j, :])
            file_.write(line)
        file_.write(line)


@pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """  Compare results from the RESTUD program and the RESPY package.
        """
        # Impose some constraints on the initialization file which ensures that the problem can
        # be solved by the RESTUD code. The code is adjusted to run with zero draws.
        constr = dict()
        constr['edu'] = (10, 20)
        constr['flag_deterministic'] = True
        constr['level'] = 0.00
        constr['types'] = 1

        # Generate random initialization file. The RESTUD code uses the same random draws for the
        # solution and simulation of the model. Thus, the number of draws is required to be less
        # or equal to the number of agents.
        init_dict = generate_random_dict(constr)

        num_agents_sim = init_dict['SIMULATION']['agents']
        num_draws_emax = init_dict['SOLUTION']['draws']
        if num_draws_emax < num_agents_sim:
            init_dict['SOLUTION']['draws'] = num_agents_sim

        # There are also some coefficients here that are not part of the original RESTUD program,
        # such as (1) separate cost of re-entry into education, (2) sheepskin effects,
        # and (3) bonus for any experience in occupation.
        init_dict['EDUCATION']['coeffs'][-1] = init_dict['EDUCATION']['coeffs'][-2]
        init_dict['OCCUPATION A']['coeffs'][-4:] = [0.0, 0.0, 0.0, 0.0]
        init_dict['OCCUPATION B']['coeffs'][-4:] = [0.0, 0.0, 0.0, 0.0]

        print_init_dict(init_dict)

        # Indicate RESTUD code the special case of zero disturbance.
        open('.restud.testing.scratch', 'a').close()

        # We need to indicate to the RESFORT code to rescale the experience covariates.
        open('.restud.respy.scratch', 'a').close()

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        # This flag aligns the random components between the RESTUD program and RESPY package.
        # The existence of the file leads to the RESTUD program to write out the random components.
        optim_paras, edu_spec, num_agents_sim, num_periods, num_draws_emax = \
            dist_class_attributes(respy_obj, 'optim_paras', 'edu_spec', 'num_agents_sim',
                'num_periods', 'num_draws_emax')

        transform_respy_to_restud(optim_paras, edu_spec, num_agents_sim, num_periods,
            num_draws_emax)

        # Solve model using RESTUD code.
        cmd = TEST_RESOURCES_DIR + '/kw_dp3asim'
        subprocess.check_call(cmd, shell=True)

        # Solve model using RESPY package.
        simulate_observed(respy_obj, is_missings=False)

        # Compare the simulated dataset generated by the programs.
        column_labels = []
        column_labels += ['Experience_A', 'Experience_B']
        column_labels += ['Years_Schooling', 'Lagged_Activity']

        py = pd.read_csv('data.respy.dat', delim_whitespace=True, header=0, na_values='.',
            usecols=column_labels).astype(np.float)

        fort = pd.DataFrame(np.array(np.genfromtxt('ftest.txt', missing_values='.'), ndmin=2)[:,
                            -4:], columns=column_labels).astype(np.float)

        # The simulated dataset from FORTRAN includes an indicator for the lagged activities.
        py['Lagged_Activity'] = py['Lagged_Activity'].map({0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0})

        assert_frame_equal(py, fort)
