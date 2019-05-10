import subprocess
from pathlib import Path


def main():
    """Run the scalability exercise.

    Define the model, a list with different number of threads and a maximum number of
    function evaluations.

    """
    model = "kw_data_one"
    num_threads = [1, 2, 4, 6, 8, 10]
    maxfun = 100

    filepath = Path(__file__).resolve().parent / "run_single_scalability_exercise.py"

    for version in ["python", "fortran"]:
        for num_thread in num_threads:
            subprocess.check_call(
                ["python", version, str(filepath), model, str(maxfun), str(num_thread)]
            )


if __name__ == "__main__":
    main()
