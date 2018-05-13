import os


def record_solution_progress(indicator, file_sim, period=None, num_states=None):

    if indicator == 1:
        if os.path.exists(file_sim + '.respy.sol'):
            os.unlink(file_sim + '.respy.sol')

        line = 'Starting state space creation'
    elif indicator == 2:
        line = 'Starting calculation of systematic rewards'
    elif indicator == 3:
        line = 'Starting backward induction procedure'
    elif indicator == 4:
        string = '''{:>18}{:>3}{:>5} {:>7} {:>7}'''
        line = string.format(*['... solving period', period, 'with', num_states, 'states'])
    elif indicator == -1:
        line = '... finished\n'
    elif indicator == -2:
        line = '... not required due to myopic agents'
    else:
        raise AssertionError

    with open(file_sim + '.respy.sol', 'a') as outfile:
        outfile.write('  ' + line + '\n\n')


def record_solution_prediction(results, file_sim):
    """ Write out some basic information to the solutions log file.
    """

    with open(file_sim + '.respy.sol', 'a') as outfile:
        outfile.write('    Information about Prediction Model')

        string = '      {:<19}' + '{:15.4f}' * 9
        outfile.write(string.format('Coefficients', *results.params))
        outfile.write(string.format('Standard Errors', *results.bse))

        string = '      {0:<19}{1:15.4f}\n'
        outfile.write(string.format('R-squared', results.rsquared))
