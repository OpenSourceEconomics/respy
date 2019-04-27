import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    """Run the estimation of a model using a number of threads and a maximum of function
    evaluations.

    Currently, we circumvent the optimization by setting maxfun to 0 and just looping
    over the estimation.

    """
    model = sys.argv[1]
    num_threads = sys.argv[2]
    maxfun = sys.argv[3]

    # Set number of threads
    if not num_threads == "-1":
        os.environ["NUMBA_NUM_THREADS"] = f"{num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"

    # Late import of respy to ensure that environment variables are read.
    import respy
    from respy import RespyCls
    from respy.python.interface import respy_interface

    # Get model
    options_spec = json.loads(
        Path(respy.__path__[0], "tests", "resources", f"{model}.json").read_text()
    )
    params_spec = pd.read_csv(
        Path(respy.__path__[0], "tests", "resources", f"{model}.csv")
    )

    # Adjust options
    options_spec["program"]["version"] = "python"
    options_spec["estimation"]["draws"] = 200
    options_spec["estimation"]["maxfun"] = 0
    options_spec["estimation"]["optimizer"] = "SCIPY-LBFGSB"
    options_spec["solution"]["draws"] = 500

    # Go into temporary folder
    folder = f"__{num_threads}"
    if Path(folder).exists():
        shutil.rmtree(folder)

    Path(folder).mkdir()
    os.chdir(folder)

    # Initialize the class
    respy_obj = RespyCls(params_spec, options_spec)

    # Simulate the data
    respy_obj, simulated_data = respy_obj.simulate()

    # Run the estimation
    start = dt.datetime.now()
    for _ in range(int(maxfun)):
        respy_interface(respy_obj, "estimate", simulated_data)
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
