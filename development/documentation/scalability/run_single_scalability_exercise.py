import datetime as dt
import json
import os
import sys


def main():
    """Evaluate the criterion function multiple times for a scalability report.

    The criterion function is evaluated ``maxfun``-times. The number of threads used is
    limited by environment variables. ``respy`` has to be imported after the environment
    variables are set as Numpy, Numba and others load them at import time.

    """
    model = sys.argv[1]
    maxfun = int(sys.argv[2])
    n_threads = int(sys.argv[3])

    # Validate input.
    assert maxfun >= 0, "Maximum number of function evaluations cannot be negative."
    assert n_threads >= 1 or n_threads == -1, (
        "Use -1 to impose no restrictions on maximum number of threads or choose a "
        "number higher than zero."
    )

    # Set number of threads
    os.environ["NUMBA_NUM_THREADS"] = f"{n_threads}"
    os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
    os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
    os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"

    # Late import of respy to ensure that environment variables are read by Numpy, etc..
    import respy as rp

    # Get model
    params, options = rp.get_example_model(model, with_data=False)

    # Simulate the data
    simulate = rp.get_simulate_func(params, options)
    df = simulate(params)

    # Get the criterion function and the parameter vector.
    crit_func = rp.get_crit_func(params, options, df)

    # Run the estimation
    start = dt.datetime.now()

    for _ in range(maxfun):
        crit_func(params)

    end = dt.datetime.now()

    # Aggregate information
    output = {
        "model": model,
        "maxfun": maxfun,
        "n_threads": n_threads,
        "start": str(start),
        "end": str(end),
        "duration": str(end - start),
    }

    # Save time to file
    with open("scalability_results.txt", "a+") as file:
        file.write(json.dumps(output))
        file.write("\n")


if __name__ == "__main__":
    main()
