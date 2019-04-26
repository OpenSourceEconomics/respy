import subprocess
from pathlib import Path


def main():
    """Run the scalability exercise.

    Define the model, a list with different number of threads and a maximum number of
    function evaluations.

    """
    model = "kw_data_one"
    num_threads = [4]
    maxfun = 1000

    filepath = Path(__file__).resolve().parent / "_run.py"

    for num_thread in num_threads:
        subprocess.check_call(
            ["python", str(filepath), model, str(num_thread), str(maxfun)]
        )


if __name__ == "__main__":
    main()
