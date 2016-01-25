""" This module contains some auxiliary functions helpful in the
quantification of ambiguity.
"""

# standard library
import pickle as pkl
import numpy as np


def store_results(rslt):
    """ Store results for further processing.
    """
    # Extract available levels of ambiguity.
    levels = list(rslt.keys())
    levels.sort()
    # Store results to file
    pkl.dump(rslt, open('rslts/quantification_ambiguity.robupy.pkl', 'wb'))
    # Open file for logging purposes.
    with open('rslts/quantification_ambiguity.robupy.log', 'w') as out_file:
        # Write out heading information.
        fmt = ' {0:<15}{1:<15}{2:<15}\n\n'
        args = ('Level', 'Value', 'Loss (in %)')
        out_file.write(fmt.format(*args))
        # Iterate over all available levels.
        for level in levels:
            # Get interesting results.
            value, baseline = rslt[level], rslt[0.0]
            difference = ((value / baseline) - 1.0) * 100
            # Format and print string.
            fmt = ' {0:<15.3f}{1:<15.3f}{2:<+15.5f}\n'
            args = (level, value, difference)
            # Write out file.
            out_file.write(fmt.format(*args))
        out_file.write('\n\n')


def distribute_arguments(parser):
    """ Distribute command line arguments.
    """
    # Process command line arguments
    args = parser.parse_args()

    # Extract arguments
    is_recompile = args.is_recompile
    num_procs = args.num_procs
    is_debug = args.is_debug
    grid = args.grid
    spec = args.spec

    # Extend levels to include baseline.
    if 0.0 not in grid:
        grid += [0.0]

    # Check arguments
    assert (is_recompile in [True, False])
    assert (is_debug in [True, False])
    assert (isinstance(num_procs, int))
    assert (num_procs > 0)
    assert (spec in ['one', 'two', 'three'])

    # Check and process information about grid.
    assert (len(grid) == 3)
    grid = float(grid[0]), float(grid[1]), int(grid[2])

    # Finishing
    return grid, is_recompile, is_debug, num_procs, spec