import numpy as np
import time

from respy.python.shared.shared_utilities import spectral_condition_number
from respy.python.shared.shared_auxiliary import cholesky_to_coeffs
from respy.python.shared.shared_auxiliary import dist_econ_paras
from respy.python.shared.shared_constants import opt_ambi_info
from respy.python.record.record_warning import record_warning
from respy.python.shared.shared_auxiliary import get_cholesky
from respy.python.shared.shared_constants import LARGE_FLOAT


def record_estimation_scaling(x_optim_free_unscaled_start,
        x_optim_free_scaled_start, paras_bounds_free_scaled,
        precond_matrix, paras_fixed):

    with open('est.respy.log', 'w') as out_file:
        out_file.write(' {:}\n\n'.format('PRECONDITIONING'))
        fmt_ = '   {:>10}' + '    {:>25}' * 5 + '\n\n'
        labels = ['Identifier', 'Original', 'Scale']
        labels += ['Transformed Value', 'Transformed Lower']
        labels += ['Transformed Upper']
        out_file.write(fmt_.format(*labels))

        j = 0
        for i in range(27):
            if paras_fixed[i]:
                continue

            paras = [i, x_optim_free_unscaled_start[j], precond_matrix[j,
                                                                       j]]
            paras += [x_optim_free_scaled_start[j]]
            paras += [paras_bounds_free_scaled[j, 0]]
            paras += [paras_bounds_free_scaled[j, 1]]

            for k in [4, 5]:
                if abs(paras[k]) > LARGE_FLOAT:
                    paras[k] = '---'
                else:
                    paras[k] = '{:25.15f}'.format(paras[k])

            fmt = '   {:>10}' + '    {:25.15f}' * 3 + '    {:>25}' * 2 + \
                  '\n'
            out_file.write(fmt.format(*paras))

            j = j + 1

        out_file.write('\n')


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


def record_estimation_eval(opt_obj, fval, x_optim_all_unscaled):
    """ Logging the progress of an estimation. This function contains two
    parts as two files provide information about the progress.
    """

    # Distribute class attributes
    paras_fixed = opt_obj.paras_fixed

    shocks_cholesky, _ = get_cholesky(x_optim_all_unscaled, 0)
    shocks_coeffs = cholesky_to_coeffs(shocks_cholesky)

    # Identify events
    is_start = (opt_obj.num_eval == 0)
    is_step = (opt_obj.crit_vals[1] > fval)

    # Update class attributes
    if is_start:
        opt_obj.crit_vals[0] = fval
        opt_obj.x_optim_container[:, 0] = x_optim_all_unscaled
        opt_obj.x_econ_container[:17, 0] = x_optim_all_unscaled[:17]
        opt_obj.x_econ_container[17:, 0] = shocks_coeffs

    if is_step:
        opt_obj.num_step += 1
        opt_obj.crit_vals[1] = fval
        opt_obj.x_optim_container[:, 1] = x_optim_all_unscaled
        opt_obj.x_econ_container[:17, 1] = x_optim_all_unscaled[:17]
        opt_obj.x_econ_container[17:, 1] = shocks_coeffs

    if True:
        opt_obj.num_eval += 1
        opt_obj.crit_vals[2] = fval
        opt_obj.x_optim_container[:, 2] = x_optim_all_unscaled
        opt_obj.x_econ_container[:17, 2] = x_optim_all_unscaled[:17]
        opt_obj.x_econ_container[17:, 2] = shocks_coeffs

    x_optim_container = opt_obj.x_optim_container
    x_econ_container = opt_obj.x_econ_container

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
            if paras_fixed[i]:
                continue
            out_file.write(
                fmt_.format(*[i, x_optim_container[i, 0],
                              x_optim_container[i, 1],
                              x_optim_container[i, 2]]))

        out_file.write('\n')

        cond = []
        for which in ['Start', 'Step', 'Current']:
            if which == 'Start':
                paras = x_econ_container[:, 0].copy()
            elif which == 'Step':
                paras = x_econ_container[:, 1].copy()
            else:
                paras = x_econ_container[:, 2].copy()

            shocks_cov = dist_econ_paras(paras)[-1]
            cond += [np.log(spectral_condition_number(shocks_cov))]
        fmt_ = '   {:>9} ' + '    {:25.15f}' * 3 + '\n'
        out_file.write(fmt_.format(*['Condition'] + cond))

        if opt_ambi_info[0] != 0:
            fmt_ = '   {:>9} ' + '    {:25.15f}\n'
            share = float(opt_ambi_info[1]) / float(opt_ambi_info[0])
            out_file.write(fmt_.format(*['Ambiguity', share]))
        else:
            fmt_ = '   {:>9} ' + '    {:>25}\n'
            out_file.write(fmt_.format(*['Ambiguity', '---']))

        out_file.write('\n')

        for i in range(3):
            if is_large[i]:
                record_warning(i + 1)

    write_est_info(0, opt_obj.crit_vals[0], x_econ_container[:, 0],
        opt_obj.num_step, opt_obj.crit_vals[1], x_econ_container[:, 1],
        opt_obj.num_eval, opt_obj.crit_vals[2], x_econ_container[:, 2])


def record_estimation_final(success, message):
    """ We summarize the results of the estimation.
    """
    with open('est.respy.log', 'a') as out_file:
        out_file.write(' ESTIMATION REPORT\n\n')
        out_file.write('   Success ' + str(success) + '\n')
        out_file.write('   Message ' + message + '\n')


def write_est_info(num_start, value_start, paras_start, num_step,
                   value_step, paras_step, num_eval, value_current, paras_current):

    # Write information to file.
    with open('est.respy.info', 'w') as out_file:
        # Write out information about criterion function
        out_file.write('\n{:>25}\n\n'.format('Criterion Function'))
        fmt_ = '{0:>25}    {1:>25}    {2:>25}    {3:>25}\n\n'
        out_file.write(fmt_.format(*['', 'Start', 'Step', 'Current']))

        line = '{:>25}'.format('')

        crit_vals = [value_start, value_step, value_current]

        is_large = [False, False, False]

        for i in range(3):
            try:
                is_large[i] = abs(crit_vals[i]) > LARGE_FLOAT
            except TypeError:
                is_large[i] = True

        for i in range(3):
            if is_large[i]:
                line += '    {:>25}'.format('---')
            else:
                line += '    {:25.15f}'.format(crit_vals[i])

        out_file.write(line + '\n\n')

        out_file.write('\n{:>25}\n\n'.format('Economic Parameters'))
        fmt_ = '{0:>25}    {1:>25}    {2:>25}    {3:>25}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Start', 'Step', 'Current']))
        fmt_ = '{0:>25}    {1:25.15f}    {2:25.15f}    {3:25.15f}\n'
        for i, _ in enumerate(range(27)):
            paras = [i, paras_start[i], paras_step[i], paras_current[i]]
            out_file.write(fmt_.format(*paras))

        fmt_ = '\n{0:<25}{1:>25}\n'
        out_file.write(fmt_.format(*[' Number of Steps', num_step]))
        out_file.write(fmt_.format(*[' Number of Evaluations', num_eval]))
