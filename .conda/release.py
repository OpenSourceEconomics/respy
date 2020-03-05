"""Build, test, convert and upload a conda package.

For the upload step to work you have to log into your anaconda.org account
before you run the script. The steps for this are explained here:
https://conda.io/docs/user-guide/tutorials/build-pkgs.html

"""
from pathlib import Path

from conda_build.api import build


REPO = (Path(__file__).parent / "..").resolve()


if __name__ == "__main__":
    build(str(REPO), user="OpenSourceEconomics", need_source_download=False)
