import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def main():
    # Set number of threads
    model = sys.argv[1]
    num_threads = sys.argv[2]
    maxfun = sys.argv[3]

    if not num_threads == "-1":
        os.environ["NUMBA_NUM_THREADS"] = f"{num_threads}"
        os.environ["MKL_NUM_THREADS"] = f"{num_threads}"
        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
        os.environ["NUMEXPR_NUM_THREADS"] = f"{num_threads}"

    # Late import of respy to ensure that environment variables are read.
    import respy
    from respy import RespyCls

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
    options_spec["estimation"]["maxfun"] = maxfun
    options_spec["estimation"]["optimizer"] = "SCIPY-BFGS"
    options_spec["preconditioning"].update(
        {"type": "identity", "minimum": 1e-5, "eps": 1e-6}
    )
    options_spec["solution"]["draws"] = 500

    # Adjust parameters
    params_spec.iloc[0, 2] = 0.95

    # Go into temp folder
    if Path("temp").exists():
        shutil.rmtree("temp")

    Path("temp").mkdir()
    os.chdir("temp")

    # Initialize the class
    respy_obj = RespyCls(params_spec, options_spec)

    # Simulate the data
    respy_obj, simulated_data = respy_obj.simulate()

    # Run the estimation
    start = dt.datetime.now()
    respy_obj.fit()
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
    shutil.rmtree("temp")

    # Save time to file
    with open("data.txt", "a+") as file:
        file.write(json.dumps(output))
        file.write("\n")


if __name__ == "__main__":
    main()
