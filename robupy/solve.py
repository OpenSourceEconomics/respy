""" This module serves as the interface to the solution functions.
"""

# project library
from robupy.fortran.solve_fortran import solve_fortran
from robupy.python.solve_python import solve_python
from robupy.simulate import simulate

''' Public function
'''


def solve(robupy_obj):
    """ Solve dynamic programming problem by backward induction.
    """
    # Antibugging
    assert (robupy_obj.get_status())

    # Distribute class attributes
    version = robupy_obj.get_attr('version')

    store = robupy_obj.get_attr('store')

    # Select appropriate interface
    if version == 'FORTRAN':

        robupy_obj = solve_fortran(robupy_obj)

    else:

        robupy_obj = solve_python(robupy_obj)

    # Set flag that object includes the solution objects.
    robupy_obj.unlock()

    robupy_obj.set_attr('is_solved', True)

    robupy_obj.lock()

    # Store results if requested
    if store:
        robupy_obj.store('solution.robupy.pkl')

    # Simulate model. The model is directly simulated in the ROBUFORT
    # executable.
    if version in ['PYTHON', 'F2PY']:
        simulate(robupy_obj)

    # Finishing
    return robupy_obj