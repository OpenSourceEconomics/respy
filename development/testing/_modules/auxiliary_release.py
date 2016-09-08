import pickle as pkl

import shutil
import sys
import pip
import os

sys.path.insert(0, '../../../../respy/tests')
from codes.random_init import generate_init


def install(version):
    """ Prepare the
    """
    cmd = ['install', '-vvv', '--no-binary', 'respy', '--extra-index-url']
    cmd += ['https://testpypi.python.org/pypi', 'respy==' + version]
    pip.main(cmd)

    # TODO: PYTEST is part of the package requirements in the newer releases.
    # So this can also be removed in the near future.
    cmd = ['install', 'pytest']
    pip.main(cmd)


def prepare_release_tests(constr):
    """ This function prepares the initialization files so that they can be
    processed by both releases under investigation. The idea is to have all
    hand-crafted modifications grouped in this function only.
    """
    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['level'] = 0.00
    constr['flag_ambiguity'] = True
    constr['file_est'] = '../data.respy.dat'

    init_dict = generate_init(constr)

    init_dict['ESTIMATION']['tau'] = int(init_dict['ESTIMATION']['tau'])
    pkl.dump(init_dict, open('new/init_dict.respy.pkl', 'wb'))

    del init_dict['AMBIGUITY']
    pkl.dump(init_dict, open('old/init_dict.respy.pkl', 'wb'))


def run_estimation():
    """ Run an estimation with the respective release.
    """
    from respy import estimate
    from respy import RespyCls

    from respy.python.shared.shared_auxiliary import print_init_dict
    init_dict = pkl.load(open('init_dict.respy.pkl', 'rb'))

    print_init_dict(init_dict)

    respy_obj = RespyCls('test.respy.ini')
    pkl.dump(estimate(respy_obj)[1], open('crit_val.respy.pkl', 'wb'))

if __name__ == '__main__':

    if sys.argv[1] == 'prepare':
        version = sys.argv[2]
        install(version)
    # TODO: This is a temporary bugfix. See main script for details.
    elif sys.argv[1] == 'upgrade':
            pip.main(['install', '--upgrade', 'pip'])
    elif sys.argv[1] == 'estimate':
        which = sys.argv[2]
        run_estimation()
    else:
        raise NotImplementedError
