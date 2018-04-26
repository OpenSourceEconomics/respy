import numpy as np
import os
from os.path import join, exists
from shutil import rmtree, copy
from time import time
from respy.python.shared.shared_auxiliary import print_init_dict
from codes.random_init import generate_init
from respy import RespyCls, estimate
from datetime import timedelta, datetime
import traceback
# import random_init
# import estimate


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
        new_dir = join(old_dir, str(seed) + '_' + t)
        os.mkdir(new_dir)
    for file in ['career_data.respy.dat', 'career_data.respy.pkl']:
        copy(join(old_dir, file), join(new_dir, file))
    os.chdir(new_dir)

    # define the constraints (constr)
    constr = {}
    constr['file_est'] = join(new_dir, 'career_data.respy.dat')
    constr['agents'] = np.random.randint(500, 1372 + 1)
    constr['edu'] = (10, np.random.randint(11, 21))
    constr['flag_estimation'] = True
    ini = generate_init(constr)
    print_init_dict(ini)
    try:
        respy_obj = RespyCls('test.respy.ini')
        estimate(respy_obj)
    except:
        tb = traceback.format_exc()
        passed = False
        error_message = str(tb)

    os.chdir(old_dir)
    if is_investigation is False:
        rmtree(new_dir)
    return passed, error_message


def run_for_hours_sequential(hours, initial_seed):
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


def run_for_hours_parallel(hours, num_procs, initial_seeds):
    raise NotImplementedError
    #return []
