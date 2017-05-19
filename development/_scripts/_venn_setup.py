import subprocess
import pip
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = PROJECT_DIR.replace('/development/_scripts', '')

# The following packages are useful during development, but are not included in the requirements
# file.
PACKAGES = ['matplotlib', 'jupyter', 'pytest-xdist']

if __name__ == '__main__':

    os.chdir(PROJECT_DIR)

    # We now install all packages that are part of the requirements file anyway.
    cmd = ['install', '-r', 'requirements.txt']
    pip.main(cmd)

    # Now we should be able to to install the package in developer node.
    cmd = [sys.executable, 'setup.py', 'develop']
    subprocess.check_call(cmd)

    # Finally we add some packages that are useful during development.
    for package in PACKAGES:
        cmd = ['install', package]
        pip.main(cmd)
