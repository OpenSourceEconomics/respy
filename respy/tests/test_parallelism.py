import numpy as np
import pytest

from respy.pre_processing.model_processing import write_init_file
from respy.python.shared.shared_constants import IS_PARALLEL

from codes.random_init import generate_random_dict
from codes.auxiliary import simulate_observed
from codes.auxiliary import compare_est_log

from respy import estimate
from respy import RespyCls


@pytest.mark.skipif(not IS_PARALLEL, reason='No PARALLELISM available')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This test ensures that it makes no difference whether the
        criterion function is evaluated in parallel or not.
        """
        # Generate random initialization file
        constr = dict()
        constr['version'] = 'FORTRAN'
        constr['maxfun'] = np.random.randint(0, 50)
        init_dict = generate_random_dict(constr)

        # If ambiguity or delta is a not fixed, we need to ensure a bound-constraint optimizer.
        # However, this is not the standard flag_estimation as the number of function evaluation
        # is possibly much larger to detect and differences in the updates of the optimizer
        # steps depending on the implementation.
        if not init_dict['BASICS']['fixed'][0]:
            init_dict['ESTIMATION']['optimizer'] = 'FORT-BOBYQA'

        base = None
        for is_parallel in [True, False]:

            init_dict['PROGRAM']['procs'] = 1
            if is_parallel:
                init_dict['PROGRAM']['procs'] = np.random.randint(2, 5)

            write_init_file(init_dict)

            respy_obj = RespyCls('test.respy.ini')
            respy_obj = simulate_observed(respy_obj)
            _, crit_val = estimate(respy_obj)

            if base is None:
                base = crit_val
            np.testing.assert_equal(base, crit_val)

    def test_2(self):
        """ This test ensures that the record files are identical.
        """
        # Generate random initialization file. The number of periods is higher than usual as only
        #  FORTRAN implementations are used to solve the random request. This ensures that also
        # some cases of interpolation are explored.
        constr = dict()
        constr['version'] = 'FORTRAN'
        constr['periods'] = np.random.randint(3, 10)
        constr['maxfun'] = 0

        init_dict = generate_random_dict(constr)

        base_sol_log, base_est_info_log = None, None
        base_est_log, base_amb_log = None, None

        for is_parallel in [False, True]:

            init_dict['PROGRAM']['procs'] = 1
            if is_parallel:
                init_dict['PROGRAM']['procs'] = np.random.randint(2, 5)

            write_init_file(init_dict)

            respy_obj = RespyCls('test.respy.ini')

            delta = respy_obj.get_attr('optim_paras')['delta']
            file_sim = respy_obj.get_attr('file_sim')

            simulate_observed(respy_obj)

            estimate(respy_obj)

            # Check for identical records
            fname = file_sim + '.respy.sol'

            if base_sol_log is None:
                base_sol_log = open(fname, 'r').read()

            op = open(fname, 'r').read()

            assert open(fname, 'r').read() == base_sol_log

            if base_est_info_log is None:
                base_est_info_log = open('est.respy.info', 'r').read()
            assert open('est.respy.info', 'r').read() == base_est_info_log

            if base_est_log is None:
                base_est_log = open('est.respy.log', 'r').readlines()
            compare_est_log(base_est_log)
