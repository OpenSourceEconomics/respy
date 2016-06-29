#!/usr/bin/env python

# standard library
import numpy as np
import time
import sys
import os

# Adding PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = ROOT_DIR.replace('/respy/python/monitoring', '')
sys.path.insert(0, ROOT_DIR)

# project library
from respy.python.estimate.estimate_auxiliary import dist_optim_paras

def update_information(num_start, value_start, paras_start, num_steps,
        value_step, paras_step, num_evals, value_current, paras_current):

    # Write information to file.
    with open('est.respy.info', 'w') as out_file:
        # Write out information about criterion function
        out_file.write('\n Criterion Function\n\n')
        fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
        out_file.write(fmt_.format(*['', 'Start', 'Step', 'Current']))
        fmt_ = '{0:>15}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n\n'
        paras = ['', value_start, value_step, value_current]

        out_file.write(fmt_.format(*paras))

        # Write out information about the optimization parameters directly.
        out_file.write('\n Optimization Parameters\n\n')
        fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Start', 'Step', 'Current']))
        fmt_ = '{0:>15}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
        for i, _ in enumerate(paras_current):
            paras = [i, paras_start[i], paras_step[i], paras_current[i]]
            out_file.write(fmt_.format(*paras))

        # Write out the current covariance matrix of the reward shocks.
        out_file.write('\n\n Covariance Matrix \n\n')

        for which in ['Start', 'Step', 'Current']:
            if which == 'Start':
                paras = paras_start
            elif which == 'Step':
                paras = paras_step
            else:
                paras = paras_current
            fmt_ = '{0:>15}   \n\n'
            out_file.write(fmt_.format(*[which]))
            shocks_cholesky = dist_optim_paras(paras, True)[-1]
            shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
            fmt_ = '{0:15.4f}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
            for i in range(4):
                out_file.write(fmt_.format(*shocks_cov[i, :]))
            out_file.write('\n')

        fmt_ = '\n{0:<25}{1:>15}\n'
        out_file.write(fmt_.format(*[' Number of Steps', num_steps]))
        out_file.write(fmt_.format(*[' Number of Evaluations', num_evals]))



def information_present():
    if not os.path.exists('est.respy.paras'):
        return False
    else:
        return True

def get_information():

    while True:

        try:
            info = np.genfromtxt('est.respy.paras')

            args = []
            args += [int(info[0, 0]), info[1, 0], info[2:, 0]]
            args += [int(info[0, 1]), info[1, 1], info[2:, 1]]
            args += [int(info[0, 2]), info[1, 2], info[2:, 2]]
            return args

        except (ValueError, IndexError) as _:
            pass


def run():

    while True:

        # Check if all required information is available. This might not be
        # the case if the optimization is still set up.
        if not information_present():
            continue

        # Collect all arguments and write out the information.
        args = get_information()

        update_information(*args)

        time.sleep(1)


if __name__ == '__main__':

    run()
