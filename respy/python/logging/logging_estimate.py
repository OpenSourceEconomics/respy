# standard library
import time

# project library
from respy.python.shared.shared_auxiliary import write_est_info
from respy.python.shared.shared_constants import LARGE_FLOAT



def log_estimation_final(opt_obj, success, message):
    fval = opt_obj.x_container[1, 1]
    with open('est.respy.log', 'a') as out_file:
        out_file.write('\n ESTIMATION REPORT\n\n')
        out_file.write('   Success ' + str(success) + '\n')
        out_file.write('   Message ' + message + '\n\n')
        fmt_ = '   {0:>9}' + '     {1:25.15f}\n'
        out_file.write(fmt_.format(*['Criterion', fval]))
        fmt_ = '\n\n   {0:>10}' + '    {1:>25}\n\n'
        out_file.write(fmt_.format(*['Identifier', 'Final']))
        fmt_ = '   {:>10}' + '    {:25.15f}\n'
        for i in range(26):
            out_file.write(
                fmt_.format(*[i, opt_obj.x_container[i + 2, 1]]))



def log_estimation_eval(opt_obj, fval):

    with open('est.respy.log', 'a') as out_file:
        fmt_ = ' {0:>4}{1:>13}' + ' ' * 10 + '{2:>4}{3:>10}\n\n'
        line = ['EVAL', opt_obj.attr['num_eval'], 'STEP', opt_obj.attr['num_step']]
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



        for i in range(26):
            out_file.write(
                fmt_.format(*[i, opt_obj.x_container[i + 2, 0],
                              opt_obj.x_container[i + 2, 1],
                              opt_obj.x_container[i + 2, 2]]))

    info_start = opt_obj.x_container[:, 0]
    info_step = opt_obj.x_container[:, 1]
    info_current = opt_obj.x_container[:, 2]



    write_est_info(int(info_start[0]), info_start[1], info_start[2:],
                   int(info_step[0]), info_step[1], info_step[2:],
                   int(info_current[0]), info_current[1], info_current[2:])
