#!/usr/bin/env python
"""Module to test the parallel implementations of the model."""
import argparse
from datetime import datetime
from datetime import timedelta

import numpy as np

from respy import RespyCls
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model


def run(hours):

    start, timeout = datetime.now(), timedelta(hours=hours)

    count = 0
    while True:
        print("COUNT", count)
        count += 1
        # Generate random initialization file
        constr = {
            "program": {"version": "fortran"},
            "estimation": {
                "maxfun": np.random.randint(0, 50),
                "optimizer": "FORT-BOBYQA",
            },
        }
        params_spec, options_spec = generate_random_model(point_constr=constr)

        base = None
        for is_parallel in [True, False]:

            if is_parallel is False:
                options_spec["program"]["threads"] = 1
                options_spec["program"]["procs"] = 1
            else:
                if IS_PARALLELISM_OMP:
                    options_spec["program"]["threads"] = np.random.randint(2, 5)
                if IS_PARALLELISM_MPI:
                    options_spec["program"]["procs"] = np.random.randint(2, 5)

            respy_obj = RespyCls(params_spec, options_spec)
            respy_obj = simulate_observed(respy_obj)
            _, crit_val = respy_obj.fit()

            if base is None:
                base = crit_val
            np.testing.assert_equal(base, crit_val)

        #  Timeout.
        if timeout < datetime.now() - start:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run parallelism test.")

    parser.add_argument(
        "--hours",
        action="store",
        dest="hours",
        help="hours to run",
        required=True,
        type=float,
    )

    args = parser.parse_args()
    hours = args.hours

    run(hours)
