import numpy as np

import shutil
import copy
import os

from respy import RespyCls
from respy import simulate

from auxiliary_economics import float_to_string


def run(spec_dict):
    """ Simulate a number of economies with varying number of economies.
    """

    # Cleanup results from a previous run and prepare the directory structure.
    if os.path.exists('rslt'):
        shutil.rmtree('rslt')
    os.mkdir('rslt')

    os.chdir('rslt')

    respy_base = RespyCls('../../../graphs.respy.ini')

    # Construct the specs for the quantification exercise.
    levels = spec_dict['levels']

    # Check that baseline either is already available or requested.
    if 0.0 not in levels:
        levels += [0.0]

    est_level = respy_base.get_attr('model_paras')['level']
    if est_level not in levels:
        levels += [float(est_level)]

    # Prepare directory structure for request.
    for level in levels:
        dirname = float_to_string(level)
        os.mkdir(dirname)

    # Update to the baseline initialization file.
    respy_base.unlock()
    for key_ in spec_dict['update'].keys():
        respy_base.set_attr(key_, spec_dict['update'][key_])
    respy_base.lock()

    for level in levels:
        respy_obj = copy.deepcopy(respy_base)
        respy_obj.attr['model_paras']['level'] = np.array([level])

        os.chdir(float_to_string(level))

        respy_obj.write_out()

        respy_obj, _ = simulate(respy_obj)

        # Back to root directory
        os.chdir('../')

    os.chdir('../')