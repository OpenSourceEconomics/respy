import os
import shutil

import numpy as np

import respy as rp
from development.modules.auxiliary_shared import get_random_dirname
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import process_model_spec
from respy.python.interface import minimal_estimation_interface
from respy.python.shared.shared_constants import DATA_FORMATS_EST
from respy.python.shared.shared_constants import DATA_LABELS_EST
from respy.python.shared.shared_constants import TOL
from respy.tests.codes.auxiliary import minimal_simulate_observed
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model


def get_chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def create_single(idx):
    """This function creates a single test."""
    dirname = get_random_dirname(5)
    os.mkdir(dirname)
    os.chdir(dirname)

    # We impose a couple of constraints that make the requests manageable.
    np.random.seed(idx)

    constr = {
        "program": {"version": "python"},
        "preconditioning": {"type": np.random.choice(["identity", "magnitudes"])},
        "estimation": {"maxfun": 0, "optimizer": "SCIPY-LBFGSB"},
    }
    constr["flag_estimation"] = True

    param_spec, options_spec = generate_random_model(point_constr=constr)
    attr = process_model_spec(param_spec, options_spec)
    df = minimal_simulate_observed(attr)
    _, crit_val = minimal_estimation_interface(attr, df)

    # In rare instances, the value of the criterion function might be too large and thus
    # printed as a string. This occurred in the past, when the gradient preconditioning
    # had zero probability observations. We now generate random initialization files
    # with smaller gradient step sizes.
    if not isinstance(crit_val, float):
        raise AssertionError(" ... value of criterion function too large.")

    # Cleanup of temporary directories.from
    os.chdir("..")
    shutil.rmtree(dirname)

    return attr, crit_val


def check_single(tests, idx):
    """This function checks a single test from the dictionary."""
    attr, crit_val = tests[idx]

    # Skip fortran.
    if attr["version"] == "fortran" or attr["maxfun"] != 0:
        return True

    # We need to create an temporary directory, so the multiprocessing does not
    # interfere with any of the files that are printed and used during the small
    # estimation request.
    dirname = get_random_dirname(5)
    os.mkdir(dirname)
    os.chdir(dirname)

    df = minimal_simulate_observed(attr)

    _, est_val = minimal_estimation_interface(attr, df)

    is_success = np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)

    # Cleanup of temporary directories.from
    os.chdir("..")
    shutil.rmtree(dirname)

    return is_success
