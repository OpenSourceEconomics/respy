#!/usr/bin/env python
""" This script compiles and executes the upgraded file that inspired the
RESPY package.
"""
import os
import shutil
from pathlib import Path


def main():
    # Compiler options. Note that the original codes are not robust enough to
    # execute in debug mode.
    DEBUG_OPTIONS = (
        " -O2  -Wall -Wline-truncation -Wcharacter-truncation "
        " -Wsurprising  -Waliasing -Wimplicit-interface  -Wunused-parameter "
        " -fwhole-file -fcheck=all  -fbacktrace -g -fmax-errors=1 "
        " -ffpe-trap=invalid,zero"
    )

    PRODUCTION_OPTIONS = " -O3"

    # I rely on production options for this script, as I also run the estimation
    # below.
    OPTIONS = PRODUCTION_OPTIONS

    # Some strings that show up repeatedly in compiler command.
    MODULES = "imsl_replacements.f90"
    LAPACK = "-L/usr/lib/lapack -llapack"

    # Copy required initialization files from the original codes.
    path = Path("..", "original")
    for fname in ["seed.txt", "in1.txt"]:
        shutil.copy(str(path / fname), ".")

    # Compiling and calling executable for simulation.
    cmd = " gfortran " + OPTIONS + " -o dp3asim " + MODULES + " dp3asim.f90 " + LAPACK
    os.system(cmd + "; ./dp3asim")

    # Compiling and calling executable for assessment of interpolation.
    for fname in ["dpsim1", "dpsim4d"]:
        cmd = (
            " gfortran "
            + OPTIONS
            + " -o "
            + fname
            + " "
            + MODULES
            + " "
            + fname
            + ".f90 "
            + LAPACK
        )
        os.system(cmd + "; ./" + fname)

    # Compiling and calling executable for estimation.
    cmd = " gfortran " + OPTIONS + " -o dpml4a " + MODULES + " dpml4a.f90 " + LAPACK
    os.system(cmd + "; ./dpml4a")


if __name__ == "__main__":
    main()
