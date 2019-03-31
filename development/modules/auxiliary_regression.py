import numpy as np
import shutil
import socket
import os

from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.pre_processing.model_processing import _params_spec_from_attributes
from respy.pre_processing.model_processing import _options_spec_from_attributes
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import TOL
from development.modules.auxiliary_shared import get_random_dirname
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_model import generate_random_model


def get_chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i: i + n]


def create_single(idx):
    """ This function creates a single test.
    """
    dirname = get_random_dirname(5)
    os.mkdir(dirname)
    os.chdir(dirname)

    # The late import is required so a potentially just compiled FORTRAN implementation
    # is recognized. This is important for the creation of the regression vault as we
    # want to include FORTRAN use cases.
    from respy import RespyCls

    # We impose a couple of constraints that make the requests manageable.
    np.random.seed(idx)

    version = np.random.choice(["python", "fortran"])

    # only choose from constraint optimizers because we always have some bounds
    if version == 'python':
        optimizer = "SCIPY-LBFGSB"
    else:
        optimizer = "FORTE-BOBYQA"

    constr = {
        "program": {"version": version},
        "preconditioning": {"type": np.random.choice(["identity", "magnitudes"])},
        "estimation": {"maxfun": int(np.random.choice(range(6), p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])),
                       "optimizer": optimizer},
    }
    constr["flag_estimation"] = True

    param_spec, options_spec = generate_random_model(point_constr=constr)
    respy_obj = RespyCls(param_spec, options_spec)
    simulate_observed(respy_obj)
    crit_val = respy_obj.fit()[1]

    # In rare instances, the value of the criterion function might be too large and thus
    # printed as a string. This occurred in the past, when the gradient preconditioning
    # had zero probability observations. We now generate random initialization files
    # with smaller gradient step sizes.
    if not isinstance(crit_val, float):
        raise AssertionError(" ... value of criterion function too large.")

    # Cleanup of temporary directories.from
    os.chdir("../")
    shutil.rmtree(dirname)

    return respy_obj.attr, crit_val


def check_single(tests, idx):
    """ This function checks a single test from the dictionary.
    """
    # Distribute test information.
    attr, crit_val = tests[idx]

    if not IS_PARALLELISM_OMP or not IS_FORTRAN:
        attr["num_threads"] = 1

    if not IS_PARALLELISM_MPI or not IS_FORTRAN:
        attr["num_procs"] = 1

    if not IS_FORTRAN:
        attr['version'] = 'python'

    # In the past we also had the problem that some of the testing machines report
    # selective failures when the regression vault was created on another machine.
    msg = " ... test is known to fail on this machine"
    if "zeus" in socket.gethostname() and idx in []:
        print(msg)
        return None
    if "acropolis" in socket.gethostname() and idx in []:
        print(msg)
        return None
    if "pontos" in socket.gethostname() and idx in []:
        print(msg)
        return None

    # We need to create an temporary directory, so the multiprocessing does not
    # interfere with any of the files that are printed and used during the small
    # estimation request.
    dirname = get_random_dirname(5)
    os.mkdir(dirname)
    os.chdir(dirname)

    # The late import is required so a potentially just compiled FORTRAN implementation
    # is recognized. This is important for the creation of the regression vault as we
    # want to include FORTRAN use cases.
    from respy import RespyCls

    params_spec = _params_spec_from_attributes(attr)
    options_spec = _options_spec_from_attributes(attr)
    respy_obj = RespyCls(params_spec, options_spec)

    simulate_observed(respy_obj)

    est_val = respy_obj.fit()[1]

    is_success = np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)

    # Cleanup of temporary directories.from
    os.chdir("../")
    shutil.rmtree(dirname)

    return is_success
