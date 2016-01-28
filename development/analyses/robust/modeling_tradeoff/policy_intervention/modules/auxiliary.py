""" This module contains some routines that help in the analysis of policy
responsiveness.
"""

# standard library
import pickle as pkl

import shlex
import os


def store_results(rslt):
    """ Store result for further processing.
    """

    # Extract information.
    levels = list(rslt.keys())
    levels.sort()
    # Write to readable file.
    with open('rslts/policy_intervention.robupy.log', 'w') as out_file:
        # Heading
        fmt = ' {0:5}{1:>15}{2:>25}\n\n'
        args = ('Level', 'Subsidy', 'Average Education')
        out_file.write(fmt.format(*args))
        for level in levels:
            subsidies = list(rslt[level].keys())
            subsidies.sort()
            for subsidy in subsidies:
                education = rslt[level][subsidy]
                fmt = ' {0:<5.4f}{1:15.2f}{2:25.7f}\n'
                args = (level, subsidy, education)
                out_file.write(fmt.format(*args))
            out_file.write('\n')

    # Persistent storage.
    pkl.dump(rslt, open('rslts/policy_intervention.robupy.pkl', 'wb'))


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_recompile = args.is_recompile
    num_procs = args.num_procs
    is_debug = args.is_debug
    spec = args.spec

    # Check arguments
    assert (num_procs > 0)
    assert (is_debug in [True, False])
    assert (is_recompile in [True, False])
    assert (spec in ['one', 'two', 'three'])

    # Finishing
    return num_procs, is_recompile, is_debug, spec


def process_models(args):
    """ This function processes the information from all simulated models.
    """
    # Switch subdirector
    os.chdir('rslts')

    # Prepare the resulting dictionary.
    rslt = dict()
    for arg in args:
        # Distribute elements of request
        _, level, subsidy = arg
        # Create required keys in dictionary
        if level not in rslt.keys():
            rslt[level] = dict()
        # Add levels
        rslt[level][subsidy] = []

    # Collect all results from the subdirectories.
    for arg in args:
        # Distribute elements of request
        _, level, subsidy = arg
        # Auxiliary objects
        name = get_name(level, subsidy)
        # Switch to results
        os.chdir(name)
        # Extract all results
        with open('data.robupy.info', 'r') as rslt_file:
            # Store results
            for line in rslt_file:
                list_ = shlex.split(line)
                try:
                    if 'Education' == list_[1]:
                        rslt[level][subsidy] = float(list_[2])
                except IndexError:
                    pass
        # Return to root directory.
        os.chdir('../../')

    # Back to root.
    os.chdir('../')

    # Finishing
    return rslt


def get_name(level, subsidy):
    """ Construct name from information about level and subsidy.
    """
    return '%03.3f' % level + '/' + '%.2f' % subsidy
