import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    """Evaluate the criterion function multiple times for a scalability report.

    The criterion function is evaluated ``maxfun``-times. The number of threads used is
    limited by environment variables. ``respy`` has to be imported after the environment
    variables are set as Numpy, Numba and others load them at import time.

    """
    model = sys.argv[1]
    maxfun = int(sys.argv[2])
    num_threads = int(sys.argv[3])

    # Validate input.
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

    # Late import of respy to ensure that environment variables are read by Numpy, etc..
    import respy as rp

    # Get model
    options, params = rp.get_example_model(model)

    # Adjust options
    options["estimation"]["maxfun"] = 0

    # Go into temporary folder
    folder = f"__{num_threads}"
    if Path(folder).exists():
        shutil.rmtree(folder)

    Path(folder).mkdir()
    os.chdir(folder)

    # Simulate the data
    state_space, simulated_data = rp.simulate(params, options)

    # Get the criterion function and the parameter vector.
    crit_func = rp.get_crit_func(params, options, simulated_data)

    # Run the estimation
    print(f"Start. Model: {model}, Maxfun: {maxfun}, Threads: {num_threads}.")
    start = dt.datetime.now()

    for _ in range(maxfun):
        crit_func(params)

    end = dt.datetime.now()

    print(f"End. Duration: {end - start} seconds.")

    # Aggregate information
    output = {
        "model": model,
        "maxfun": maxfun,
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
