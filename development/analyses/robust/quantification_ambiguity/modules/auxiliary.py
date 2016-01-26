""" This module contains some auxiliary functions helpful in the
quantification of ambiguity.
"""

# standard library
import shlex


def get_total_value():
    """ Process get value of criterion function.
    """
    with open('quantification_ambiguity.robupy.log', 'r') as output_file:
        for line in output_file.readlines():
            # Split lines
            list_ = shlex.split(line)
            # Skip all irrelevant lines.
            if not len(list_) == 3:
                continue
            if not list_[0] == 'Total':
                continue
            # Finishing
            return float(list_[2])


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