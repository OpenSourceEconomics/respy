import pickle as pkl

import shlex
import os

from auxiliary_economics import get_float_directories
from auxiliary_economics import float_to_string

GRID_DIR = '../grid/rslt'
QUANT_DIR = '../../quantification'


def run():
    """ Quantify the level of ambiguity for the different parameter values.
    """
    os.chdir(GRID_DIR)

    levels = get_float_directories()
    levels.sort()

    for level in levels:
        run_level(level)

    os.chdir('../../quantification')

    aggregate()


def run_level(level):
    """ Simulate the economy for a given level of ambiguity.
    """
    os.chdir(float_to_string(level))

    # Get the EMAX for further processing and extract relevant information.
    respy_obj = pkl.load(open('solution.respy.pkl', 'rb'))
    total_value = respy_obj.get_attr('periods_emax')[0, 0]

    # Store results for aggregation.
    with open('quantification.respy.log', 'a') as file_:
        string = '    {:<15}     {:>15.4f}'*2
        args = ['Level', level, 'Total Value', total_value]
        file_.write(string.format(*args))

    os.chdir('../')


def aggregate():
    """ Aggregate results by processing the log files.
    """
    os.chdir(GRID_DIR)

    # Get all directories that are contain information about the criterion
    # function for selected pairs of ambiguity and psychic costs.
    levels = get_float_directories()
    levels.sort()
    # Iterate over all ambiguity levels and intercepts.
    rslts = dict()
    for level in levels:
        os.chdir(float_to_string(level))
        rslts[level] = []
        # Another process might still be running.
        if os.path.exists('quantification.respy.log'):
            crit = get_total_value()
            rslts[level] = crit
        # Ready for a new candidate intercept.
        os.chdir('../')

    os.chdir('../../quantification')

    # Open file for logging purposes.
    with open('quantification.respy.log', 'w') as out_file:
        # Write out heading information.
        fmt = ' {0:>15}{1:>15}{2:>15}\n\n'
        args = ('Level', 'Value', 'Loss (in %)')
        out_file.write(fmt.format(*args))
        # Iterate over all available levels.
        for level in levels:
            # Get interesting results.
            value, baseline = rslts[level], rslts[0.0]
            difference = ((value / baseline) - 1.0) * 100
            # Format and print string.
            fmt = ' {0:>15.4f}{1:>15.3f}{2:>+15.5f}\n'
            args = (level, value, difference)
            # Write out file.
            out_file.write(fmt.format(*args))
        out_file.write('\n\n')
    # Store results for further processing.
    pkl.dump(rslts, open('quantification.respy.pkl', 'wb'))
    # Back to root directory.
    os.chdir('../')


def get_total_value():
    """ Get value of criterion function.
    """
    with open('quantification.respy.log', 'r') as output_file:
        for line in output_file.readlines():
            list_ = shlex.split(line)
            return float(list_[4])
