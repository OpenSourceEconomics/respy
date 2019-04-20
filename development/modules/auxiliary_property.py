import fileinput
import glob
import importlib
import os
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# RESPY directory. This allows to compile_ the debug version of the FORTRAN
# program.
RESPY_DIR = os.path.dirname(os.path.realpath(__file__))
RESPY_DIR = RESPY_DIR.replace("development/testing/property/modules", "") + "respy"

PYTHON_EXEC = sys.executable

from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import IS_F2PY


def compile_package():
    """ Compile RESPY package in debug mode.
    """
    current_directory = os.getcwd()
    os.chdir(RESPY_DIR)
    os.system(
        PYTHON_EXEC + " waf distclean; " + PYTHON_EXEC + " waf configure build --debug"
    )
    os.chdir(current_directory)


def finalize_testing_record(full_test_record):
    """ Add the temporary file with the information about tracebacks to the
    main report.
    """

    # Count total and failed tests.
    total_tests, failed_tests = 0, 0
    for module in full_test_record.keys():
        for method in full_test_record[module].keys():
            total_tests += sum(full_test_record[module][method])
            failed_tests += full_test_record[module][method][1]

    # Indicate that test run is finished
    with open("property.respy.info", "a") as log_file:
        log_file.write("\n\tRUN COMPLETED\n\n")
        fmt_ = "\t\t{0[0]:<15}{0[1]:>9}\n\n"
        log_file.write(fmt_.format(["TOTAL TESTS", total_tests]))
        log_file.write(fmt_.format(["FAILED TESTS", failed_tests]))


def initialize_record_canvas(full_test_record, start, timeout):
    """ This function initializes the raw log file.
    """
    start_time = start.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start + timeout).strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("property.respy.info", "w") as log_file:
        # Write out some header information.
        log_file.write("\n\n")
        str_ = "\t{0[0]:<15}{0[1]:<20}\n\n"
        log_file.write(str_.format(["START", start_time]))
        log_file.write(str_.format(["FINISH", end_time]))
        log_file.write(str_.format(["UPDATE", current_time]))

        # Note the Python version used during execution.
        str_ = "\t{0[0]:<15}{0[1]}.{0[2]}.{0[3]}\n\n"
        log_file.write(str_.format(["PYTHON"] + list(sys.version_info[:3])))

        log_file.write("\n\n")
        # Iterate over all modules. There is a potential conflict in the
        # namespace.
        for module_ in full_test_record.keys():
            str_ = "\t{0[0]:<27}{0[1]:<20}{0[2]:<20} \n\n"
            log_file.write(str_.format([module_, "Success", "Failure"]))
            # Iterate over all methods in the particular module.
            for method_ in sorted(full_test_record[module_]):
                str_ = "\t\t{0[0]:<25}{0[1]:<20}{0[2]:<20} \n"
                success, failure = full_test_record[module_][method_]
                log_file.write(str_.format([method_, success, failure]))

            log_file.write("\n\n")

        log_file.write("-" * 79)
        log_file.write("\n" + "-" * 79)


def update_testing_record(module, method, seed_test, is_success, msg, full_test_record):
    """ Maintain a record about the outcomes of the testing efforts.
    """
    # Formatting of time objects.

    str_ = "\t\t{0[0]:<25}{0[1]:<20}{0[2]:<20} \n"
    success, failure = full_test_record[module][method]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_line = "\t{0[0]:<15}{0[1]:<20}\n\n".format(["UPDATE", current_time])
    is_module, is_method, is_update = False, False, False

    for line in fileinput.input("property.respy.info", inplace=True):

        list_ = shlex.split(line)
        # Skip empty lines
        if len(list_) > 0:
            if list_[0] == module:
                is_module = True
                is_method = False

            if list_[0] == method:
                is_method = True

            is_update = list_[0] == "UPDATE"

        if is_method and is_module or is_update:
            if is_update:
                new_line = update_line
            else:
                new_line = str_.format([method, success, failure])
            print(new_line.rstrip())

            is_module, is_update = False, False
        else:
            print(line.rstrip())

    # Append Error message
    if not is_success:
        # Write out the traceback message to file for future inspection.
        with open("property.respy.info", "a") as log_file:
            str_ = "\nMODULE {0[0]:<25} METHOD {0[1]:<25} SEED: {0[" "2]:<10} \n\n"
            log_file.write(str_.format([module, method, seed_test]))
            log_file.write(msg)
            log_file.write("\n" + "-" * 79 + "\n\n")


def get_test_dict(test_dir):
    """ This function constructs a dictionary with the recognized test
    modules as the keys. The corresponding value is a list with all test
    methods inside that module.
    """
    # Get all test modules
    test_modules = test_dir.glob("*.py")
    # Add test_dir to path for importing
    sys.path.append(str(test_dir))

    # Given the modules, get all tests methods.
    test_dict = {}
    for test_module in test_modules:
        if test_module.stem == "__init__":
            continue
        test_dict[test_module.stem] = []
        mod = importlib.import_module(test_module.stem)
        candidate_methods = dir(mod.TestClass)
        for candidate_method in candidate_methods:
            if "test_" in candidate_method:
                test_dict[test_module.stem].append(candidate_method)

    # Remove path from PYTHONPATH
    sys.path.remove(str(test_dir))

    # If the PARALLELISM or FORTRAN is not available, we remove the parallel tests.
    if not IS_PARALLELISM_MPI and not IS_PARALLELISM_OMP:
        del test_dict["test_parallelism"]

    if not IS_FORTRAN:
        del test_dict["test_versions"]
        del test_dict["test_restud"]
        test_dict["test_integration"].remove("test_9")

    if not IS_F2PY:
        del test_dict["test_f2py"]

    return test_dict


def get_random_request(test_dict):
    """ This function extracts a random request from the dictionary.
    """
    # Create a list of tuples. Each tuple denotes a unique combination of
    # a module and a test contained in it.
    candidates = []
    for key_ in sorted(test_dict.keys()):
        for value in test_dict[key_]:
            candidates.append((key_, value))
    # Draw a random combination.
    index = np.random.randint(0, len(candidates))
    module, method = candidates[index]

    # Finishing
    return module, method


def cleanup_testing_infrastructure(keep_results, keep_dataset=False):
    """ This function cleans up before and after a testing run. If requested,
    the log file is retained.
    """
    files = glob.glob("*")
    for fname in files:
        if ".py" in fname or ".pbs" in fname:
            continue
        if keep_results:
            if "property.respy.info" in fname:
                continue
        if keep_dataset:
            if "career_data.respy" in fname:
                continue
        if os.path.isdir(fname):
            shutil.rmtree(fname)
        elif os.path.isfile(fname):
            os.unlink(fname)
        else:
            pass
