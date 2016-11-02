from scipy.optimize.slsqp import _minimize_slsqp
from scipy.optimize import approx_fprime
from scipy.optimize import rosen_der
from scipy.optimize import rosen

import numpy as np
import pytest
import sys

from respy.python.solve.solve_ambiguity import construct_emax_ambiguity
from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.python.shared.shared_constants import TEST_RESOURCES_DIR
from respy.python.solve.solve_ambiguity import constraint_ambiguity
from respy.python.solve.solve_ambiguity import criterion_ambiguity
from respy.python.shared.shared_auxiliary import print_init_dict
from respy.python.solve.solve_ambiguity import get_worst_case
from respy.python.solve.solve_ambiguity import kl_divergence
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import TEST_DIR

from codes.auxiliary import simulate_observed
from codes.random_init import generate_init


from respy import RespyCls
from respy import estimate

# Edit of PYTHONPATH required for PYTHON 2 as no __init__.py in tests
# subdirectory. If __init__.py is added, the path resolution for PYTEST
# breaks down.
if IS_FORTRAN:
    sys.path.insert(0, TEST_RESOURCES_DIR)
    import f2py_interface as fort_debug

sys.path.insert(0, TEST_DIR)

from test_integration import TestClass as link_integration
from test_versions import TestClass as link_versions
from test_f2py import TestClass as link_f2py


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This test ensures that using the ambiguity functionality with a
        tiny level yields the same results as using the risk functionality
        directly.

        """
        seed_observed = np.random.randint(0, 100)

        constr = dict()
        constr['maxfun'] = 0
        constr['flag_parallelism'] = False

        init_dict = generate_init(constr)

        base_val = None
        for level in [0.00, 0.00000000001]:

            init_dict['AMBIGUITY']['coeffs'] = [level]
            init_dict['AMBIGUITY']['bounds'] = [[0.00, None]]

            print_init_dict(init_dict)

            respy_obj = RespyCls('test.respy.ini')

            simulate_observed(respy_obj)
            _, crit_val = estimate(respy_obj)

            if base_val is None:
                base_val = crit_val

            np.testing.assert_allclose(base_val, crit_val)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_2(self):
        """ Testing the F2PY implementations of the ambiguity-related functions.
        """
        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'
        constraints['flag_ambiguity'] = True

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        respy_obj = RespyCls('test.respy.ini')

        respy_obj = simulate_observed(respy_obj)

        # Extract class attributes
        periods_rewards_systematic, states_number_period, mapping_state_idx, \
            periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
            edu_max, delta, measure, model_paras, derivatives, \
            optimizer_options, file_sim = \
                dist_class_attributes(respy_obj,
                    'periods_rewards_systematic', 'states_number_period',
                    'mapping_state_idx', 'periods_emax', 'num_periods',
                    'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                    'delta', 'measure', 'model_paras', 'derivatives',
                    'optimizer_options', 'file_sim')

        level = model_paras['level']
        shocks_cholesky = model_paras['shocks_cholesky']
        shocks_cov = np.matmul(shocks_cholesky, shocks_cholesky.T)

        fort_slsqp_maxiter = optimizer_options['FORT-SLSQP']['maxiter']
        fort_slsqp_ftol = optimizer_options['FORT-SLSQP']['ftol']
        fort_slsqp_eps = optimizer_options['FORT-SLSQP']['eps']

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic rewards
        rewards_systematic = periods_rewards_systematic[period, k, :]

        args = (num_periods, num_draws_emax, period, k, draws_standard,
            rewards_systematic, edu_max, edu_start, periods_emax, states_all,
            mapping_state_idx, delta, shocks_cov, measure, level)

        py = construct_emax_ambiguity(*args + (optimizer_options,
                                               file_sim, False))
        f90 = fort_debug.wrapper_construct_emax_ambiguity(*args +
                (fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps,
                 file_sim, False))
        np.testing.assert_allclose(py, f90)

        x = np.random.uniform(-1, 1, size=2)

        args = (num_periods, num_draws_emax, period, k, draws_standard,
            rewards_systematic, edu_max, edu_start, periods_emax, states_all,
            mapping_state_idx, delta)

        py = criterion_ambiguity(x, *args)
        f90 = fort_debug.wrapper_criterion_ambiguity(x, *args)
        np.testing.assert_allclose(py, f90)

        py = approx_fprime(x, criterion_ambiguity, fort_slsqp_eps, *args)
        f90 = fort_debug.wrapper_criterion_ambiguity_derivative(x, *args + (
            fort_slsqp_eps, ))
        np.testing.assert_allclose(py, f90)

        args = (shocks_cov, level)
        py = constraint_ambiguity(x, *args)
        f90 = fort_debug.wrapper_constraint_ambiguity(x, *args)
        np.testing.assert_allclose(py, f90)

        py = approx_fprime(x, constraint_ambiguity, fort_slsqp_eps, *args)
        f90 = fort_debug.wrapper_constraint_ambiguity_derivative(x, *args +
                (fort_slsqp_eps, ))
        np.testing.assert_allclose(py, f90)

        args = (num_periods, num_draws_emax, period, k,
            draws_standard, rewards_systematic, edu_max, edu_start,
            periods_emax, states_all, mapping_state_idx, delta, shocks_cov,
            level)

        py, _, _ = get_worst_case(*args + (optimizer_options, ))
        f90, _, _ = fort_debug.wrapper_get_worst_case(*args + (
            fort_slsqp_maxiter, fort_slsqp_ftol, fort_slsqp_eps))
        np.testing.assert_allclose(py, f90)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_3(self):
        """ Check the calculation of the KL divergence across implementations.
        """
        for i in range(1000):

            num_dims = np.random.randint(1, 5)
            old_mean = np.random.uniform(size=num_dims)
            new_mean = np.random.uniform(size=num_dims)

            cov = np.random.random((num_dims, num_dims))
            old_cov = np.matmul(cov.T, cov)

            cov = np.random.random((num_dims, num_dims))
            new_cov = np.matmul(cov.T, cov)

            np.fill_diagonal(new_cov, new_cov.diagonal() * 5)
            np.fill_diagonal(old_cov, old_cov.diagonal() * 5)

            args = (old_mean, old_cov, new_mean, new_cov)

            py = kl_divergence(*args)
            fort = fort_debug.wrapper_kl_divergence(*args)

            np.testing.assert_almost_equal(fort, py)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_4(self):
        """ Check the SLSQP implementations.
        """
        # Test the FORTRAN codes against the PYTHON implementation. This is
        # expected to fail sometimes due to differences in precision between the
        # two implementations. In particular, as updating steps of the optimizer
        # are very sensitive to just small differences in the derivative
        # information. The same functions are available as a FORTRAN
        # implementations.
        def debug_constraint_derivative(x):
            return np.ones(len(x))

        def debug_constraint_function(x):
            return np.sum(x) - 10.0

        # Sample basic test case
        ftol = np.random.uniform(0.000000, 1e-5)
        maxiter = np.random.randint(1, 100)
        num_dim = np.random.randint(2, 4)

        x0 = np.random.normal(size=num_dim)

        # Evaluation of Rosenbrock function. We are using the FORTRAN version
        # in the development of the optimization routines.
        f90 = fort_debug.debug_criterion_function(x0, num_dim)
        py = rosen(x0)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        py = rosen_der(x0)
        f90 = fort_debug.debug_criterion_derivative(x0, len(x0))
        np.testing.assert_allclose(py, f90[:-1], rtol=1e-05, atol=1e-06)

        # Setting up PYTHON SLSQP interface for constraints
        constraint = dict()
        constraint['type'] = 'eq'
        constraint['args'] = ()
        constraint['fun'] = debug_constraint_function
        constraint['jac'] = debug_constraint_derivative

        # Evaluate both implementations
        fort = fort_debug.wrapper_slsqp_debug(x0, maxiter, ftol, num_dim)
        py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
                ftol=ftol, constraints=constraint)['x']

        np.testing.assert_allclose(py, fort, rtol=1e-05, atol=1e-06)

    """ This section reproduces the tests from test_versions but runs them
    with ambiguity.
    """
    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_versions_1(self, flag_ambiguity=True):
        link_versions().test_1(flag_ambiguity)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_versions_2(self, flag_ambiguity=True):
        link_versions().test_2(flag_ambiguity)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_versions_4(self, flag_ambiguity=True):
        link_versions().test_4(flag_ambiguity)

    """ This section reproduces the tests from test_f2py but runs them with
    ambiguity.
    """
    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_f2py_4(self, flag_ambiguity=True):
        link_f2py().test_4(flag_ambiguity)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_f2py_5(self, flag_ambiguity=True):
        link_f2py().test_5(flag_ambiguity)

    @pytest.mark.skipif(not IS_FORTRAN, reason='No FORTRAN available')
    def test_f2py_6(self, flag_ambiguity=True):
        link_f2py().test_6(flag_ambiguity)

    """ This section reproduces the tests from test_integration but runs them
    with ambiguity.
    """
    def test_integration_3(self, flag_ambiguity=True):
        link_integration().test_3(flag_ambiguity)

    def test_integration_4(self, flag_ambiguity=True):
        link_integration().test_4(flag_ambiguity)

    def test_integration_6(self, flag_ambiguity=True):
        link_integration().test_6(flag_ambiguity)

    def test_integration_7(self, flag_ambiguity=True):
        link_integration().test_6(flag_ambiguity)
