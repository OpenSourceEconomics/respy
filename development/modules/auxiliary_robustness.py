import numpy as np
import os
from os.path import join, exists
from shutil import rmtree, copy
from time import time
from respy.tests.codes.random_model import generate_random_model
from respy import RespyCls
from datetime import timedelta, datetime
import traceback
from functools import partial
from multiprocessing import Pool

# import random_init


def run_robustness_test(seed, is_investigation):
    """Run a single robustness test."""
    passed = True
    error_message = None
    np.random.seed(seed)
    old_dir = os.getcwd()
    t = str(time())[-6:]
    if is_investigation is True:
        new_dir = join(old_dir, str(seed))
        if exists(new_dir):
            rmtree(new_dir)
        os.mkdir(new_dir)
    else:
        new_dir = join(old_dir, str(seed) + "_" + t)
        os.mkdir(new_dir)
    for file in ["career_data.respy.dat", "career_data.respy.pkl"]:
        copy(join(old_dir, file), join(new_dir, file))
    os.chdir(new_dir)

    # We need to impose some constraints so that the random initialization file does meet the
    # structure of the empirical dataset. We need to be particularly careful with the
    # construction of the maximum level of schooling as we need to rule out that anyone in the
    # estimation sample has a value larger then the specified maximum value.
    version = np.random.choice(["python", "fortran"])
    if version == 'python':
        max_periods = 3
    else:
        max_periods = 10

    num_periods = np.random.randint(1, max_periods)
    agents = np.random.randint(500, 1372 + 1)
    edu_start = np.random.choice(range(7, 12))

    constr = {
        "num_periods": num_periods,
        "edu_spec": {
            "start": [int(edu_start)],
            "max": np.random.randint(edu_start + num_periods, 30)},
        "estimation": {"file": "career_data.respy.dat",
                       "agents": agents,
                       "maxfun": np.random.randint(1, 5)
                       },
        "program": {"version": version}
    }

    if version == 'fortran':
        constr['estimation']['optimizer'] = 'FORT-BOBYQA'

    params_spec, options_spec = generate_random_model(point_constr=constr)

    try:
        respy_obj = RespyCls(params_spec, options_spec)
        respy_obj.fit()
    except:
        tb = traceback.format_exc()
        passed = False
        error_message = str(tb)

    os.chdir(old_dir)
    if is_investigation is False:
        rmtree(new_dir)
    return passed, error_message


def run_for_hours_sequential(initial_seed, hours):
    np.random.seed(initial_seed)
    failed_dict = {}
    start = datetime.now()
    counter = 0
    timeout = timedelta(hours=hours)
    while timeout >= (datetime.now() - start):
        counter += 1
        seed = np.random.randint(1, 100000)
        passed, error_message = run_robustness_test(seed, is_investigation=False)
        if passed is False:
            failed_dict[seed] = error_message
    return failed_dict, counter


def run_for_hours_parallel(initial_seeds, hours):
    num_procs = len(initial_seeds)
    with Pool(num_procs) as p:
        run_sequential = partial(run_for_hours_sequential, hours=hours)
        result_list = p.map(run_sequential, initial_seeds)

    failed_dict = {}
    counter = 0
    for fd, c in result_list:
        failed_dict.update(fd)
        counter += c

    return failed_dict, counter
