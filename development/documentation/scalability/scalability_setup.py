import subprocess
from pathlib import Path


def main():
    """Run the scalability exercise.

    Define the model, a list with different number of threads and a maximum number of
    function evaluations.

    """
    model = "kw_data_one"
    maxfun = 100

    filepath = Path(__file__).resolve().parent / "run_single_scalability_exercise.py"

    # Run Python
    for num_thread in [1, 2, 4, 6, 8, 10]:
        subprocess.check_call(
            [
                "python",
                str(filepath),
                "python",
                model,
                str(maxfun),
                "0",
                str(num_thread),
            ]
        )

    for num_proc, num_thread in [(1, 1), (1, 2), (1, 4), (1, 6), (1, 8), (1, 10)]:
        subprocess.check_call(
            [
                "python",
                str(filepath),
                "fortran",
                model,
                str(maxfun),
                str(num_proc),
                str(num_thread),
            ]
        )


if __name__ == "__main__":
    main()
