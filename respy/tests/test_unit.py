from functools import partial
from pathlib import Path

import numpy as np
from pandas.testing import assert_series_equal

from respy import RespyCls
from respy.pre_processing.model_processing import _read_options_spec
from respy.pre_processing.model_processing import _read_params_spec
from respy.python.shared.shared_auxiliary import distribute_parameters
from respy.python.shared.shared_auxiliary import get_optim_paras
from respy.python.shared.shared_constants import DECIMALS
from respy.tests.codes.random_model import generate_random_model


assert_almost_equal = partial(np.testing.assert_almost_equal, decimal=DECIMALS)


class TestClass(object):
    """This class groups together some tests."""

    def test_1(self):
        """ Testing whether back-and-forth transformation have no effect.
        """
        for _ in range(10):
            num_types = np.random.randint(1, 5)
            num_paras = 53 + (num_types - 1) * 6

            # Create random parameter vector
            base = np.random.uniform(size=num_paras)

            x = base.copy()

            # Apply numerous transformations
            for _ in range(10):
                optim_paras = distribute_parameters(x, is_debug=True)
                args = (optim_paras, num_paras, "all", True)
                x = get_optim_paras(*args)

            np.testing.assert_allclose(base, x)

    def test_2(self):
        """ Testing whether the back and forth transformation works.
        """
        for _ in range(100):
            params_spec, options_spec = generate_random_model()
            # Process request and write out again.
            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj.write_out("alt_respy")

            new_params_spec = _read_params_spec(Path("alt_respy.csv"))
            new_options_spec = _read_options_spec(Path("alt_respy.json"))

            assert options_spec == new_options_spec

            for col in params_spec.columns:
                assert_series_equal(params_spec[col], new_params_spec[col])
