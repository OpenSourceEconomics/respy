import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    """Run the estimation of a model using a number of threads and a maximum of function
    evaluations.

    Currently, we circumvent the optimization by setting maxfun to 0 and just looping
    over the estimation.

    """
    model = sys.argv[1]
    maxfun = int(sys.argv[2])
    num_threads = int(sys.argv[3])

    # Test commandline input
    assert maxfun >= 0, "Maximum number of function evaluations cannot be negative."
    assert num_threads >= 1 or num_threads == -1, (
        "Use -1 to impose no restrictions on maximum number of threads or choose a "
        "number higher than zero."
    )

    # Set number of threads
    if not num_threads == -1:
        os.environ["NUMBA_NUM_THREADS"] = f"{num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"

    # Late import of respy to ensure that environment variables are read.
    from respy import RespyCls, get_example_models
    from respy.python.interface import respy_interface
    from respy.python.shared.shared_auxiliary import dist_class_attributes
    from respy.python.estimate.estimate_python import pyth_criterion
    from respy.python.shared.shared_auxiliary import create_draws
    from respy.python.shared.shared_auxiliary import get_optim_paras

    # Get model
    options_spec, params_spec = get_example_models(model)

    # Adjust options
    options_spec["program"]["version"] = "python"

    # Go into temporary folder
    folder = f"__{num_threads}"
    if Path(folder).exists():
        shutil.rmtree(folder)

    Path(folder).mkdir()
    os.chdir(folder)

    # Initialize the class
    respy_obj = RespyCls(params_spec, options_spec)

    # Simulate the data
    state_space, simulated_data = respy_interface(respy_obj, "simulate")

    # Create necessary inputs to pyth_criterion
    (
        optim_paras,
        num_periods,
        is_debug,
        num_draws_prob,
        seed_prob,
        num_draws_emax,
        seed_emax,
        is_interpolated,
        num_points_interp,
        tau,
        num_paras,
    ) = dist_class_attributes(
        respy_obj,
        "optim_paras",
        "num_periods",
        "is_debug",
        "num_draws_prob",
        "seed_prob",
        "num_draws_emax",
        "seed_emax",
        "is_interpolated",
        "num_points_interp",
        "tau",
        "num_paras",
    )
    periods_draws_prob = create_draws(num_periods, num_draws_prob, seed_prob, is_debug)
    periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax, is_debug)
    x_optim_all_unscaled_start = get_optim_paras(
        optim_paras, num_paras, "all", is_debug
    )

    # Run the estimation
    start = dt.datetime.now()
    for _ in range(maxfun):
        # Change parameters only a bit as result caching might be a problem. Changes in
        # parameters are only positive.
        slightly_changed_parameters = (
            x_optim_all_unscaled_start
            + np.random.uniform(low=0, high=1, size=x_optim_all_unscaled_start.shape)
            * 1e-07
        )

        pyth_criterion(
            slightly_changed_parameters,
            is_interpolated,
            num_points_interp,
            is_debug,
            simulated_data,
            tau,
            periods_draws_emax,
            periods_draws_prob,
            state_space,
        )
    end = dt.datetime.now()

    # Aggregate information
    output = {
        "model": model,
        "maxfun": maxfun,
        "num_threads": num_threads,
        "start": str(start),
        "end": str(end),
        "duration": str(end - start),
    }

    # Step out of temp folder
    os.chdir("..")

    # Save time to file
    with open("data.txt", "a+") as file:
        file.write(json.dumps(output))
        file.write("\n")


if __name__ == "__main__":
    main()
