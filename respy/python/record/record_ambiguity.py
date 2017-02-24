import numpy as np

from respy.python.shared.shared_constants import MISSING_FLOAT


def record_ambiguity(opt_ambi_details, states_number_period, num_periods,
        file_sim):
    """ Write result of optimization problem to log file.
    """

    with open(file_sim + '.respy.amb', 'a') as file_:
        for period in range(num_periods - 1, -1, -1):

            for k in range(states_number_period[period]):

                x_shift = opt_ambi_details[period, k, :2]
                div, success, mode = opt_ambi_details[period, k, 2:]

                # We need to skip states that were not analyzed during the
                # interpolation routine.
                if mode == MISSING_FLOAT:
                    continue

                message = get_message(mode)

                string = ' PERIOD{0[0]:>7}  STATE{0[1]:>7}\n\n'
                file_.write(string.format([period, k]))

                string = '   {:<12}{:10.5f}\n'
                args = ['Occupation A'] + [x_shift[0]]
                file_.write(string.format(*args))

                string = '   {:<12}{:10.5f}\n\n'
                args = ['Occupation B'] + [x_shift[1]]
                file_.write(string.format(*args))

                string = '   {:<12}{:>10.5f}\n\n'
                file_.write(string.format(*['Divergence', div]))

                string = '   {:<15}{:<5}\n'
                file_.write(string.format(*['Success', str(success == 1)]))

                string = '   {:<15}{:<100}\n\n\n'
                file_.write(string.format(*['Message', message]))

        # Write out summary information in the end to get an overall sense of
        # the performance.
        file_.write(' SUMMARY\n\n')

        string = '''{0[0]:>10} {0[1]:>10} {0[2]:>10} {0[3]:>10}\n'''
        args = string.format(['Period', 'Total', 'Success', 'Failure'])
        file_.write(args)

        file_.write('\n')

        for period in range(num_periods - 1, -1, -1):
            total = states_number_period[period]
            success = np.sum(opt_ambi_details[period, :total, 3] == 1)
            failure = np.sum(opt_ambi_details[period, :total, 3] == 0)
            success /= float(total)
            failure /= float(total)

            string = '''{0[0]:>10} {0[1]:>10} {0[2]:10.2f} {0[3]:10.2f}\n'''
            file_.write(string.format([period, total, success, failure]))

        file_.write('\n')


def get_message(mode):
    """ This function transfers the mode returned from the solver into the
    corresponding message.
    """

    if mode == -1:
        message = 'Gradient evaluation required (g & a)'
    elif mode == 0:
        message = 'Optimization terminated successfully'
    elif mode == 1:
        message = 'Function evaluation required (f & c)'
    elif mode == 2:
        message = 'More equality constraints than independent variables'
    elif mode == 3:
        message = 'More than 3*n iterations in LSQ subproblem'
    elif mode == 4:
        message = 'Inequality constraints incompatible'
    elif mode == 5:
        message = 'Singular matrix E in LSQ subproblem'
    elif mode == 6:
        message = 'Singular matrix C in LSQ subproblem'
    elif mode == 7:
        message = 'Rank-deficient equality constraint subproblem HFTI'
    elif mode == 8:
        message = 'Positive directional derivative for linesearch'
    elif mode == 9:
        message = 'Iteration limit exceeded'

    # The following are project-specific return codes.
    elif mode == 15:
        message = 'No random variation in shocks'
    elif mode == 16:
        message = 'Optimization terminated successfully'
    else:
        raise AssertionError

    return message