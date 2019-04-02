#!/usr/bin/env python
"""This module just starts numerous estimations on our empirical data.
The goal is to ensure that the code handles it all well. This increases the
robustness of the package as the data is not so well-behaved as simulations.
"""

from development.modules.auxiliary_shared import send_notification
from development.modules.auxiliary_robustness import run_robustness_test
from development.modules.auxiliary_robustness import run_for_hours_sequential
from development.modules.auxiliary_robustness import run_for_hours_parallel
from development.modules.auxiliary_career_decision_data import prepare_dataset
import argparse
import numpy as np
from pathlib import Path
from development.modules.auxiliary_property import (
    cleanup_testing_infrastructure,
)


def run(request, is_compile, is_background, num_procs, keep_dataset):
    cleanup_testing_infrastructure(
        keep_results=False, keep_dataset=keep_dataset
    )
    data_path = Path.cwd() / "career_data.respy.dat"
    if not data_path.exists():
        prepare_dataset()
    if request[0] == "investigate":
        is_investigation, is_run = True, False
    elif request[0] == "run":
        is_investigation, is_run = False, True
    else:
        raise NotImplementedError("request in [run, investigate]")

    seed_investigation, hours = None, 0.0
    if is_investigation:
        seed_investigation = int(request[1])
        assert isinstance(seed_investigation, int)
    elif is_run:
        hours = float(request[1])
        assert hours > 0.0

    if is_investigation is True:
        # run a single test with args['seed']
        passed, error_message = run_robustness_test(
            seed_investigation, is_investigation
        )
        if passed is True:
            failed_dict = {}
        else:
            failed_dict = {seed_investigation: error_message}
            print(error_message)
    else:
        # make seed list
        if num_procs == 1:
            initial_seed = np.random.randint(1, 100000)
            failed_dict, num_tests = run_for_hours_sequential(
                initial_seed=initial_seed, hours=hours
            )
            failed_seeds = list(failed_dict.keys())
        else:
            initial_seeds = np.random.randint(1, 100000, size=num_procs)
            failed_dict, num_tests = run_for_hours_parallel(
                initial_seeds=initial_seeds, hours=hours
            )

    failed_seeds = list(failed_dict.keys())

    failed = bool(failed_seeds)
    filepath = "robustness.respy.info"
    with open(filepath, "w") as file:
        file.write("Summary of Failed Robustness Tests")
        file.write("\n\n")
        if failed is True:
            for seed, message in failed_dict.items():
                file.write(str(seed))
                file.write("\n\n")
                file.write(message)
                file.write("\n\n\n\n\n")
        else:
            file.write("All tests passed.")

    if is_investigation is False:
        send_notification(
            "robustness",
            is_failed=failed,
            failed_seeds=failed_seeds,
            hours=hours,
            procs=num_procs,
            num_tests=num_tests,
        )
        cleanup_testing_infrastructure(
            keep_results=False, keep_dataset=keep_dataset
        )


if __name__ == "__main__":
    # args = process_command_line_arguments('robustness')
    # run(args)
    parser = argparse.ArgumentParser(
        description="Run or investigate robustness tests."
    )

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        help="task to perform",
        required=True,
        nargs=2,
    )

    parser.add_argument(
        "--background",
        action="store_true",
        dest="is_background",
        default=False,
        help="background process",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        dest="is_compile",
        default=False,
        help="compile RESPY package",
    )

    parser.add_argument(
        "--procs",
        action="store",
        dest="num_procs",
        default=1,
        type=int,
        help="number of processors",
    )

    parser.add_argument(
        "--keep_dataset",
        action="store_true",
        dest="keep_dataset",
        default=False,
        help="Do not generate dataset if it already exists.",
    )

    args = parser.parse_args()
    request, is_compile = args.request, args.is_compile

    if is_compile:
        raise AssertionError("... probably not working due to reload issues.")

    is_background = args.is_background
    num_procs = args.num_procs
    keep_dataset = args.keep_dataset

    run(request, is_compile, is_background, num_procs, keep_dataset)
