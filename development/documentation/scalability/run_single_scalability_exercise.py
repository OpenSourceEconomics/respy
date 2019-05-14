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
    version = sys.argv[1]
    model = sys.argv[2]
    maxfun = int(sys.argv[3])
    num_procs = int(sys.argv[4])
    num_threads = int(sys.argv[5])

    # Test commandline input
    assert maxfun >= 0, "Maximum number of function evaluations cannot be negative."
    assert num_threads >= 1 or num_threads == -1, (
        "Use -1 to impose no restrictions on maximum number of threads or choose a "
        "number higher than zero."
    )

    # Set number of threads
    if not num_threads == -1 and version == "python":
        os.environ["NUMBA_NUM_THREADS"] = f"{num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"

    # Late import of respy to ensure that environment variables are read.
    from respy import RespyCls, get_example_model
    from respy.python.interface import respy_interface
    from respy.fortran.interface import resfort_interface

    # Get model
    options_spec, params_spec = get_example_model(model)

    # Adjust options
    options_spec["program"]["version"] = version
    options_spec["estimation"]["maxfun"] = 0
    if version == "fortran":
        options_spec["program"]["procs"] = num_procs
        options_spec["program"]["threads"] = num_threads

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

    # Run the estimation
    print(
        f"Start. Program: {version}, Model: {model}, Maxfun: {maxfun}, Procs: "
        f"{num_procs}, Threads: {num_threads}."
    )
    start = dt.datetime.now()

    for _ in range(maxfun):
        if version == "python":
            respy_interface(respy_obj, "estimate", simulated_data)
        else:
            resfort_interface(respy_obj, "estimate", simulated_data)

    end = dt.datetime.now()

    print(f"End. Duration: {end - start} seconds.")

    # Aggregate information
    output = {
        "version": version,
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
