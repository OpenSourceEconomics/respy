import subprocess
from pathlib import Path


def main():
    model = "kw_data_one"
    num_threads = [1, 2, 4, 6, 8, 10]
    maxfun = 5

    filepath = Path(__file__).resolve().parent / "_run.py"

    for num_thread in num_threads:
        subprocess.check_call(
            ["python", str(filepath), model, str(num_thread), str(maxfun)]
        )


if __name__ == "__main__":
    main()
