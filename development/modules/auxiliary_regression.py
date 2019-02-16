import numpy as np
import shutil
import socket
import os

from respy.python.shared.shared_constants import IS_PARALLELISM_OMP
from respy.python.shared.shared_constants import IS_PARALLELISM_MPI
from respy.pre_processing.model_processing import write_init_file
from respy.python.shared.shared_constants import IS_FORTRAN
from respy.python.shared.shared_constants import TOL
from development.modules.auxiliary_shared import get_random_dirname
from respy.tests.codes.auxiliary import simulate_observed
from respy.tests.codes.random_init import generate_init


def get_chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


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
    constr = dict()
    constr["precond_type"] = np.random.choice(["identity", "magnitudes"])
    constr["flag_estimation"] = True

    init_dict = generate_init(constr)
    respy_obj = RespyCls("test.respy.ini")
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

    return init_dict, crit_val


def check_single(tests, idx):
    """ This function checks a single test from the dictionary.
    """
    # Distribute test information.
    init_dict, crit_val = tests[idx]

    # TODO: These are temporary modifications that ensure compatibility over time and
    # will be removed once we update the regression test battery.
    init_dict["EDUCATION"]["lagged"] = []
    for edu_start in init_dict["EDUCATION"]["start"]:
        if edu_start >= 10:
            init_dict["EDUCATION"]["lagged"] += [1.0]
        else:
            init_dict["EDUCATION"]["lagged"] += [0.0]

    init_dict["PROGRAM"]["threads"] = 1
    if IS_PARALLELISM_OMP and init_dict["PROGRAM"]["version"] == "FORTRAN":
        init_dict["PROGRAM"]["threads"] = np.random.randint(1, 5)

    # During development it is useful that we can only run the PYTHON versions of the
    # program.
    msg = " ... skipped as required version of package not available"
    if init_dict["PROGRAM"]["version"] == "FORTRAN" and not IS_FORTRAN:
        print(msg)
        return None
    if init_dict["PROGRAM"]["procs"] > 1 and not IS_PARALLELISM_MPI:
        print(msg)
        return None
    if init_dict["PROGRAM"]["threads"] > 1 and not IS_PARALLELISM_OMP:
        print(msg)
        return None

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

    write_init_file(init_dict)
    respy_obj = RespyCls("test.respy.ini")
    simulate_observed(respy_obj)

    est_val = respy_obj.fit()[1]

    is_success = np.isclose(est_val, crit_val, rtol=TOL, atol=TOL)

    # Cleanup of temporary directories.from
    os.chdir("../")
    shutil.rmtree(dirname)

    return is_success
