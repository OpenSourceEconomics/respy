
""" This modules contains some additional tests that are only used in
long-run development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal
from scipy.optimize import approx_fprime
from scipy.optimize import rosen_der
from scipy.optimize import rosen
import pandas as pd
import numpy as np
import sys
import os

# testing library
from modules.auxiliary import compile_package, transform_robupy_to_restud, \
    write_disturbances

# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.tests.random_init import generate_random_dict, print_random_dict
from robupy.python.py.ambiguity import _divergence
from robupy import read, solve, simulate


''' Main
'''
def test_96():
    """ Compare results between FORTRAN and PYTHON of selected
    hand-crafted functions. In test_97() we test FORTRAN implementations
    against PYTHON intrinsic routines.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(1000):

        # Draw random request for testing purposes
        matrix = (np.random.multivariate_normal(np.zeros(4), np.identity(
            4), 4))
        cov = np.dot(matrix, matrix.T)
        x = np.random.rand(2)
        level = np.random.random(1)
        eps = np.random.rand()**2

        # Kullback-Leibler (KL) divergence
        py = _divergence(x, cov, level)
        f90 = fort.wrapper_divergence(x, cov, level)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Gradient approximation of KL divergence
        py = approx_fprime(x, _divergence, eps, cov, level)
        f90 = fort.wrapper_divergence_approx_gradient(x, cov, level, eps)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)


def test_97():
    """ Compare results between FORTRAN and PYTHON of selected functions. The
    file python/f2py/debug_interface.f90 provides the F2PY bindings.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    import robupy.python.f2py.f2py_debug as fort

    for _ in range(1000):

        # Draw random requests for testing purposes.
        num_draws = np.random.random_integers(2, 1000)
        dim = np.random.random_integers(1, 6)
        mean = np.random.uniform(-0.5, 0.5, (dim))

        matrix = (np.random.multivariate_normal(np.zeros(dim), np.identity(dim), dim))
        cov = np.dot(matrix, matrix.T)

        # Inverse
        py = np.linalg.inv(cov)
        f90 = fort.wrapper_inverse(cov, dim)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Determinant
        py = np.linalg.det(cov)
        f90 = fort.wrapper_determinant(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Trace
        py = np.trace(cov)
        f90 = fort.wrapper_trace(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Cholesky decomposition
        f90 = fort.wrapper_cholesky(cov, dim)
        py = np.linalg.cholesky(cov)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Random normal deviates. This only tests the interface, requires
        # visual inspection in IPYTHON notebook as well.
        fort.wrapper_standard_normal(num_draws)
        fort.wrapper_multivariate_normal(mean, cov, num_draws, dim)

        # Evaluation of Rosenbrock function. We are using the FORTRAN version
        # in the development of the optimization routines.
        x0 = np.random.randn(num_draws)

        f90 = fort.wrapper_rosenbrock(x0, num_draws)
        py = rosen(x0)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        py = rosen_der(x0)
        f90 = fort.wrapper_rosenbrock_derivative(x0, len(x0))
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

def test_98():
    """  Compare results from the RESTUD program and the ROBUPY package.
    """

    # Ensure that fast solution methods are available
    compile_package('fast')

    # Prepare RESTUD program
    os.chdir('modules')
    os.system(' gfortran -fcheck=bounds -o dp3asim dp3asim.f95 >'
              ' /dev/null 2>&1 ')
    os.remove('pei_additions.mod')
    os.remove('imsl_replacements.mod')
    os.chdir('../')

    # Impose some constraints on the initialization file which ensures that
    # the problem can be solved by the RESTUD code.
    constraints = dict()
    constraints['edu'] = (10, 20)
    constraints['level'] = 0.00
    constraints['debug'] = 'True'

    version = np.random.choice(['FORTRAN', 'F2PY', 'PYTHON'])
    constraints['version'] = version

    # Generate random initialization file. The RESTUD code uses the same random
    # draws for the solution and simulation of the model. Thus, the number of
    # draws is required to be less or equal to the number of agents.
    init_dict = generate_random_dict(constraints)

    num_agents = init_dict['SIMULATION']['agents']
    num_draws = init_dict['SOLUTION']['draws']
    if num_draws < num_agents:
        init_dict['SOLUTION']['draws'] = num_agents

    print_random_dict(init_dict)

    # Write out disturbances to align the three implementations.
    write_disturbances(init_dict)

    # Perform toolbox actions
    robupy_obj = read('test.robupy.ini')

    # This flag aligns the random components between the RESTUD program and
    # ROBUPY package. The existence of the file leads to the RESTUD program
    # to write out the random components.
    init_dict = robupy_obj.get_attr('init_dict')

    transform_robupy_to_restud(init_dict)

    # Solve model using RESTUD code.
    os.system('./modules/dp3asim > /dev/null 2>&1')

    # Solve model using ROBUPY package.
    solve(robupy_obj)

    # Compare the simulated datasets generated by the programs.
    py = pd.DataFrame(np.array(np.genfromtxt('data.robupy.dat',
            missing_values='.'), ndmin=2)[:, -4:])

    fort = pd.DataFrame(np.array(np.genfromtxt('ftest.txt',
            missing_values='.'), ndmin=2)[:, -4:])

    assert_frame_equal(py, fort)

    # Cleanup
    os.unlink('disturbances.txt')


def test_99():
    """ Testing whether the results from a fast and slow execution of the
    code result in identical simulate datasets.
    """
    # Ensure that fast solution methods are available
    compile_package('fast')

    # Constraint to risk model
    constraints = dict()
    constraints['level'] = 0.0
    constraints['debug'] = 'True'

    # Just making sure that it also works for this special case.
    if np.random.choice([True, False]):
        constraints['eps_zero'] = True

    # Generate random initialization file. Since this exercise is only for
    # debugging purposes, the codes uses the same disturbances for the
    # simulation and solution of the model. Thus, the number of agents cannot
    # be larger than the number of draws.
    init_dict = generate_random_dict(constraints)

    num_agents = init_dict['SIMULATION']['agents']
    num_draws = init_dict['SOLUTION']['draws']
    if num_draws < num_agents:
        init_dict['SOLUTION']['draws'] = num_agents

    print_random_dict(init_dict)

    # Write out disturbances to align the three implementations.
    write_disturbances(init_dict)

    # Initialize containers
    base = None

    for version in ['PYTHON', 'F2PY', 'FORTRAN']:

        # Prepare initialization file
        init_dict['PROGRAM']['version'] = version

        print_random_dict(init_dict)

        # Simulate the ROBUPY package
        os.system('robupy-solve --model test.robupy.ini')

        # Load simulated data frame
        data_frame = pd.read_csv('data.robupy.dat', delim_whitespace=True)

        data_frame.to_csv(version + '.dat')

        # Compare
        if base is None:
            base = data_frame.copy()

        assert_frame_equal(base, data_frame)