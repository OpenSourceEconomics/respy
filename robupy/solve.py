""" This module serves as the interface to the solution functions.
"""

# project library
from robupy.fortran.solve_fortran import solve_fortran
from robupy.python.solve_python import solve_python

''' Public function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    version = robupy_obj.get_attr('version')

    # Select appropriate interface
    if version == 'FORTRAN':

        robupy_obj = solve_fortran(robupy_obj)

    else:

        robupy_obj = solve_python(robupy_obj)

    # Finishing
    return robupy_obj