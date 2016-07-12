from respy.python.shared.shared_auxiliary import add_solution
from respy.python.solve.solve_auxiliary import check_input
from respy.fortran.interface import resfort_interface
from respy.python.interface import respy_interface


def solve(respy_obj):
    """ Solve the model
    """
    # Checks, cleanup, start logger
    assert check_input(respy_obj)

    # Distribute class attributes
    is_store = respy_obj.get_attr('is_store')
    version = respy_obj.get_attr('version')

    # Select appropriate interface.
    if version == 'FORTRAN':
        solution = resfort_interface(respy_obj, 'solve')
    elif version == 'PYTHON':
        solution = respy_interface(respy_obj, 'solve')
    else:
        raise NotImplementedError

    # Attach solution to class instance
    respy_obj = add_solution(respy_obj, *solution)

    respy_obj.unlock()
    respy_obj.set_attr('is_solved', True)
    respy_obj.lock()

    # Store object to file
    if is_store:
        respy_obj.store('solution.respy.pkl')

    # Finishing
    return respy_obj


