""" This module serves as the interface to the performance library.
"""


def get_library(fast):
    """ Load the applicable set of functions
    """
    # Antibugging
    assert (fast in [True, False])

    # Load library
    if fast:
        import robupy.performance.fortran.fortran_functions as perf_lib
    else:
        import robupy.performance.python.python_functions as perf_lib

    # Finishing
    return perf_lib