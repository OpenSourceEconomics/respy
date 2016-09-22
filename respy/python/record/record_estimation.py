import numpy as np
import time

from respy.python.shared.shared_auxiliary import dist_optim_paras
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_constants import LARGE_FLOAT


def record_estimation_scalability(which):

    fmt_ = '   {:<6}     {:>10}     {:>8}\n'

    today = time.strftime("%d/%m/%Y")
    now = time.strftime("%H:%M:%S")

    if which == 'Start':
        with open('.scalability.respy.log', 'w') as out_file:
            out_file.write(fmt_.format(*[which, today, now]))
    elif which == 'Finish':
        with open('.scalability.respy.log', 'a') as out_file:
            out_file.write(fmt_.format(*[which, today, now]))
    else:
        raise AssertionError


def record_estimation_stop():
    with open('est.respy.info', 'a') as out_file:
        out_file.write('\n TERMINATED\n')


def record_estimation_eval(opt_obj, fval):
    """ Logging the progress of an estimation. This function contains two
    parts as two files provide information about the progress.
    """

    # Now we turn to est.respy.info
    with open('est.respy.log', 'a') as out_file:
        fmt_ = ' {0:>4}{1:>13}' + ' ' * 10 + '{2:>4}{3:>10}\n\n'
        line = ['EVAL', opt_obj.num_eval, 'STEP', opt_obj.num_step]
        out_file.write(fmt_.format(*line))
        fmt_ = '   {0:<9}     {1:>25}\n'
        out_file.write(fmt_.format(*['Date', time.strftime("%d/%m/%Y")]))
        fmt_ = '   {0:<9}     {1:>25}\n'
        out_file.write(fmt_.format(*['Time', time.strftime("%H:%M:%S")]))

        if abs(fval) < LARGE_FLOAT:
            fmt_ = '   {0:>9}     {1:25.15f}\n\n'
            out_file.write(fmt_.format(*['Criterion', fval]))
        else:
            fmt_ = '   {0:>9}     {1:>25}\n\n'
            out_file.write(fmt_.format(*['Criterion', '---']))

        fmt_ = '   {:>10}' + '    {:>25}' * 3 + '\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Start', 'Step', 'Current']))

        fmt_ = '   {:>10}' + '    {:25.15f}' * 3 + '\n'

        value_start = opt_obj.crit_vals[0]
        value_current = opt_obj.crit_vals[2]

        is_large = [False, False, False]
        is_large[0] = abs(value_start) > LARGE_FLOAT
        is_large[1] = abs(opt_obj.crit_vals[1]) > LARGE_FLOAT
        is_large[2] = abs(value_current) > LARGE_FLOAT

        for i in range(27):
            out_file.write(
                fmt_.format(*[i, opt_obj.x_container[i, 0],
                    opt_obj.x_container[i, 1],
                              opt_obj.x_container[i, 2]]))

        out_file.write('\n')

        for i in range(3):
            if is_large[i]:
                record_warning(i + 1)

    write_est_info(0, opt_obj.crit_vals[0], opt_obj.x_container[:, 0],
        opt_obj.num_step, opt_obj.crit_vals[1], opt_obj.x_container[:, 1],
        opt_obj.num_eval, opt_obj.crit_vals[2], opt_obj.x_container[:, 2])


def record_estimation_final(opt_obj, success, message):
    """ We summarize the results of the estimation.
    """
    fval = opt_obj.crit_vals[1]
    with open('est.respy.log', 'a') as out_file:
        out_file.write(' ESTIMATION REPORT\n\n')
        out_file.write('   Success ' + str(success) + '\n')
        out_file.write('   Message ' + message + '\n\n')

        fmt_ = '   {0:>9}' + '     {1:45.15f}\n'
        out_file.write(fmt_.format(*['Criterion', fval]))
        fmt_ = '\n\n   {0:>10}' + '    {1:>25}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Final']))
        fmt_ = '   {:>10}' + '    {:25.5f}\n'
        for i in range(27):
            out_file.write(
                fmt_.format(*[i, opt_obj.x_container[i, 1]]))
        out_file.write('\n')


def write_est_info(num_start, value_start, paras_start, num_step,
                   value_step, paras_step, num_eval, value_current, paras_current):

    # Write information to file.
    with open('est.respy.info', 'w') as out_file:
        # Write out information about criterion function
        out_file.write('\n Criterion Function\n\n')
        fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
        out_file.write(fmt_.format(*['', 'Start', 'Step', 'Current']))

        line = '{:>15}'.format('')

        crit_vals = [value_start, value_step, value_current]

        is_large = [False, False, False]

        for i in range(3):
            try:
                is_large[i] = abs(crit_vals[i]) > LARGE_FLOAT
            except TypeError:
                is_large[i] = True

        for i in range(3):
            if is_large[i]:
                line += '    {:>15}'.format('---')
            else:
                line += '    {:15.4f}'.format(crit_vals[i])

        out_file.write(line + '\n\n')

        # Write out information about the optimization parameters directly.
        out_file.write('\n Optimization Parameters\n\n')
        fmt_ = '{0:>15}    {1:>15}    {2:>15}    {3:>15}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Start', 'Step', 'Current']))
        fmt_ = '{0:>15}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
        for i, _ in enumerate(paras_current):
            paras = [i, paras_start[i], paras_step[i], paras_current[i]]
            out_file.write(fmt_.format(*paras))

        # Transform the optimization parameter to level units.
        out_file.write('\n')
        paras = ['Level'] + [paras_start[0] ** 2, paras_step[0]  ** 2,
                             paras_current[0] ** 2]
        out_file.write(fmt_.format(*paras))


        # Write out the current covariance matrix of the reward shocks.
        out_file.write('\n\n Covariance Matrix\n\n')

        for which in ['Start', 'Step', 'Current']:
            if which == 'Start':
                paras = paras_start
            elif which == 'Step':
                paras = paras_step
            else:
                paras = paras_current
            fmt_ = '{0:>15}\n\n'
            out_file.write(fmt_.format(*[which]))
            shocks_cholesky = dist_optim_paras(paras, True)[-1]
            shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)
            fmt_ = '{0:15.4f}    {1:15.4f}    {2:15.4f}    {3:15.4f}\n'
            for i in range(4):
                out_file.write(fmt_.format(*shocks_cov[i, :]))
            out_file.write('\n')

        fmt_ = '\n{0:<25}{1:>15}\n'
        out_file.write(fmt_.format(*[' Number of Steps', num_step]))
        out_file.write(fmt_.format(*[' Number of Evaluations', num_eval]))
