import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    """Run the estimation of a model using a number of threads and a maximum of function
    evaluations.

    Currently, we circumvent the optimization by setting maxfun to 0 and just looping
    over the estimation.

    """
    model = sys.argv[1]
    maxfun = int(sys.argv[2])
    num_procs = int(sys.argv[3])
    num_threads = int(sys.argv[4])

    # Test commandline input
    assert maxfun >= 0, "Maximum number of function evaluations cannot be negative."
    assert num_threads >= 1 or num_threads == -1, (
        "Use -1 to impose no restrictions on maximum number of threads or choose a "
        "number higher than zero."
    )

    # Set number of threads
    os.environ["NUMBA_NUM_THREADS"] = f"{num_threads}"
    os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
    os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"

    # Late import of respy to ensure that environment variables are read.
    import respy as rp
    from respy.interface import (
        minimal_estimation_interface,
        minimal_simulation_interface,
    )
    from respy.pre_processing.model_processing import process_model_spec

    # Get model
    options_spec, params_spec = rp.get_example_model(model)

    # Adjust options
    options_spec["estimation"]["maxfun"] = 0

    # Go into temporary folder
    folder = f"__{num_threads}"
    if Path(folder).exists():
        shutil.rmtree(folder)

    Path(folder).mkdir()
    os.chdir(folder)

    # Initialize the class
    attr = process_model_spec(params_spec, options_spec)

    # Simulate the data
    state_space, simulated_data = minimal_simulation_interface(attr)

    # Run the estimation
    print(
        f"Start. Model: {model}, Maxfun: {maxfun}, Procs: {num_procs}, "
        f"Threads: {num_threads}."
    )
    start = dt.datetime.now()

    for _ in range(maxfun):
        minimal_estimation_interface(attr, simulated_data)

    end = dt.datetime.now()

    print(f"End. Duration: {end - start} seconds.")

    # Aggregate information
    output = {
        "model": model,
        "maxfun": maxfun,
        "num_procs": num_procs,
        "num_threads": num_threads,
        "start": str(start),
        "end": str(end),
        "duration": str(end - start),
    }

    # Step out of temp folder and delete it
    os.chdir("..")
    shutil.rmtree(folder)

    # Save time to file
    with open("scalability_results.txt", "a+") as file:
        file.write(json.dumps(output))
        file.write("\n")


if __name__ == "__main__":
    main()
