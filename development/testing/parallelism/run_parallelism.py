#!/usr/bin/env python
"""Module to test the parallel implementations of the model."""
import argparse
from datetime import timedelta
from datetime import datetime
import numpy as np

from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.pre_processing.model_processing import write_init_file

from respy.tests.codes.random_init import generate_random_dict
from respy.tests.codes.auxiliary import simulate_observed

from respy import RespyCls


def run(hours):

    start, timeout = datetime.now(), timedelta(hours=hours)

    count = 0
    while True:
        print("COUNT", count)
        count += 1
        # Generate random initialization file
        constr = dict()
        constr["version"] = "FORTRAN"
        constr["maxfun"] = np.random.randint(0, 50)
        init_dict = generate_random_dict(constr)

        # We fix an optimizer that is always valid.
        init_dict["ESTIMATION"]["optimizer"] = "FORT-BOBYQA"

        base = None
        for is_parallel in [True, False]:

            init_dict["PROGRAM"]["threads"] = 1
            init_dict["PROGRAM"]["procs"] = 1

            if is_parallel:
                if IS_PARALLELISM_OMP:
                    init_dict["PROGRAM"]["threads"] = np.random.randint(2, 5)
                if IS_PARALLELISM_MPI:
                    init_dict["PROGRAM"]["procs"] = np.random.randint(2, 5)

            write_init_file(init_dict)

            respy_obj = RespyCls("test.respy.ini")
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
