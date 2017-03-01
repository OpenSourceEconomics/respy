#!/usr/bin/env python
""" I will now try to run some estimations.
"""

import os
import sys

if len(sys.argv) > 1:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('git clean -d -f; ./waf configure build --debug --without_f2py') \
           == 0
    os.chdir(cwd)




import shutil

import time


from respy.python.shared.shared_auxiliary import print_init_dict

import numpy as np
from respy.python.solve.solve_ambiguity import criterion_ambiguity, \
    get_worst_case, construct_emax_ambiguity


from respy import RespyCls
from respy import simulate
from respy import estimate
from codes.auxiliary import simulate_observed

from codes.random_init import generate_init
from respy.python.shared.shared_auxiliary import dist_class_attributes

from codes.auxiliary import write_draws

np.random.seed(123)
respy_obj = RespyCls('model.respy.ini')

# base_crit = None
# for num_procs in [1]:
#     version = 'procs' + str(num_procs)
#     if os.path.exists(version):
#         shutil.rmtree(version)
#
#     #os.mkdir(version)
#     #os.chdir(version)
#     #shutil.copy('../draws.txt', 'draws.txt')
#     respy_obj.reset()
#     respy_obj.unlock()
#     respy_obj.attr['num_procs'] = num_procs
#     respy_obj.lock()
#     respy_obj = simulate_observed(respy_obj)
#     _, crit = estimate(respy_obj)
    #if base_crit is None:
    #    base_crit = crit
    #np.testing.assert_almost_equal(crit, base_crit)

    #os.chdir('../')
    #print(crit)

# Generate constraint periods
constr = dict()
constr['version'] = 'PYTHON'
constr['flag_estimation'] = True
constr['flag_ambiguity'] = True

for i in range(100):
    seed = i + 1
    np.random.seed(seed)
    print("seed ", seed)

    # Generate random initialization file
    init_dict = generate_init(constr)
    respy_obj = RespyCls('test.respy.ini')
    respy_obj = simulate_observed(respy_obj)
    _, crit = estimate(respy_obj)
    os.system('git clean -d -f')
    print(crit)


#
#     init_dict['AMBIGUITY']['mean'] = False
#     print_init_dict(init_dict)
#     respy_obj = RespyCls('test.respy.ini')
#
#     respy_obj = simulate_observed(respy_obj)
#
#     # Extract class attributes
#     periods_rewards_systematic, states_number_period, mapping_state_idx, periods_emax, num_periods, states_all, num_draws_emax, edu_start, edu_max, ambi_spec, optim_paras, optimizer_options, file_sim = dist_class_attributes(
#         respy_obj, 'periods_rewards_systematic', 'states_number_period',
#         'mapping_state_idx', 'periods_emax', 'num_periods', 'states_all',
#         'num_draws_emax', 'edu_start', 'edu_max', 'ambi_spec', 'optim_paras',
#         'optimizer_options', 'file_sim')
#
#     max_states_period = max(states_number_period)
# #    shocks_cov = np.matmul(optim_paras['shocks_cholesky'],
# #        optim_paras['shocks_cholesky'].T)
#
#     dim = 4
#     matrix = np.random.uniform(size=dim ** 2).reshape(dim, dim)
#     shocks_cov = np.dot(matrix, matrix.T)
#
#
#     # Sampling of random period and admissible state index
#     period = np.random.choice(range(num_periods))
#     k = np.random.choice(range(states_number_period[period]))
#
#     # Select systematic rewards
#     rewards_systematic = periods_rewards_systematic[period, k, :]
#
#     # Sample draws
#     draws_standard = np.random.multivariate_normal(np.zeros(4), np.identity(4),
#         (num_draws_emax,))
#     x0 = np.tile(0.0, 2)
#     if not ambi_spec['mean']:
#         rho_start = shocks_cov[0, 1] / (
#         np.sqrt(shocks_cov[0, 0]) * np.sqrt(shocks_cov[1, 1]))
#         x0 = np.append(x0,
#             [np.sqrt(shocks_cov[0, 0]), np.sqrt(shocks_cov[1, 1])])
#
#     else:
#         x0 = np.tile(0.0, (2))
#     #
#     # criterion_ambiguity(x0, num_periods, num_draws_emax, period, k,
#     #      draws_standard,
#     #     rewards_systematic, edu_max, edu_start,
#     #         periods_emax, states_all, mapping_state_idx, optim_paras, shocks_cov)
#     #
#     draws_emax_standard = draws_standard
#     # get_worst_case(num_periods, num_draws_emax, period, k, draws_emax_standard,
#     #     rewards_systematic, edu_max, edu_start, periods_emax, states_all,
#     #     mapping_state_idx, shocks_cov, optim_paras, optimizer_options,
#     #     ambi_spec)
#
#     i, j = num_periods, max_states_period
#     opt_ambi_details = np.tile(-99.0, (i, j, 7))
#
#
#     ambi_spec = dict()
#     ambi_spec['mean'] = False
#     ambi_spec['measure'] = 'abs'
#
# #    shocks_cov = np.zeros((4, 4))
#
#     construct_emax_ambiguity(num_periods, num_draws_emax, period, k,
#         draws_emax_standard, rewards_systematic, edu_max, edu_start,
#         periods_emax, states_all, mapping_state_idx, shocks_cov, ambi_spec,
#         optim_paras, optimizer_options, opt_ambi_details)
#
#     #print('going in')
#     #x, base =
#     #print(base)
#     #np.testing.assert_almost_equal(0.350116964137, base)
#
