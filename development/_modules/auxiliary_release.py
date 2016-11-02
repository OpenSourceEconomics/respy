import pickle as pkl

import shutil
import shlex
import sys
import pip
import os


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
    # This script is also imported (but not used) for the creation of the
    # virtual environments. Thus, the imports might not be valid when
    # starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['level'] = 0.00
    constr['fixed_ambiguity'] = True
    constr['file_est'] = '../data.respy.dat'

    init_dict = generate_init(constr)

    # In the old release, there was just one location to define the step size
    # for all derivative approximations.
    eps = np.round(init_dict['SCIPY-BFGS']['eps'], decimals=15)
    init_dict['PRECONDITIONING']['eps'] = eps
    init_dict['FORT-BFGS']['eps'] = eps

    # We did not have any preconditioning implemented in the PYTHON version
    # initially. We had to switch the preconditioning scheme in the new
    # release and now use the absolute value and thus preserve the sign of
    # the derivative.
    init_dict['PRECONDITIONING']['type'] = 'identity'

    # Some of the optimization algorithms were not available in the old release.
    opt_pyth = np.random.choice(['SCIPY-BFGS', 'SCIPY-POWELL'])
    opt_fort = np.random.choice(['FORT-BFGS', 'FORT-NEWUOA'])

    if init_dict['PROGRAM']['version'] == 'PYTHON':
        init_dict['ESTIMATION']['optimizer'] = opt_pyth
    else:
        init_dict['ESTIMATION']['optimizer'] = opt_fort

    del init_dict['FORT-BOBYQA']
    del init_dict['SCIPY-LBFGSB']

    # The concept of bounds for parameters was not available and the
    # coefficients in the initialization file were only printed to the first
    # four digits.
    for label in ['HOME', 'OCCUPATION A', 'OCCUPATION B', 'EDUCATION', 'SHOCKS']:
        num = len(init_dict[label]['fixed'])
        coeffs = np.round(init_dict[label]['coeffs'], decimals=4)
        init_dict[label]['bounds'] = [(None, None)] * num
        init_dict[label]['coeffs'] = coeffs

    # In the original release we treated TAU as an integer when printing to
    # file by accident.
    init_dict['ESTIMATION']['tau'] = int(init_dict['ESTIMATION']['tau'])
    pkl.dump(init_dict, open('new/init_dict.respy.pkl', 'wb'))

    # Now we just turn to to restructuring the old initialization dictionary
    # so it can be properly processed.
    init_dict['SHOCKS']['fixed'] = np.array(init_dict['SHOCKS']['fixed'])

    # Added more fine grained scaling. Needs to be aligned across old/new
    # with identity or flag False first and then we want to allow for more
    # nuanced check.
    init_dict['SCALING'] = dict()
    init_dict['SCALING']['flag'] = (init_dict['PRECONDITIONING']['type'] == 'gradient')
    init_dict['SCALING']['minimum'] = init_dict['PRECONDITIONING']['minimum']

    # More flexible parallelism. We removed the extra section onn
    # parallelism.
    init_dict['PARALLELISM'] = dict()
    init_dict['PARALLELISM']['flag'] = init_dict['PROGRAM']['procs'] > 1
    init_dict['PARALLELISM']['procs'] = init_dict['PROGRAM']['procs']

    # We had a section that enforced the same step size for the derivative
    # calculation in each.
    init_dict['DERIVATIVES'] = dict()
    init_dict['DERIVATIVES']['version'] = 'FORWARD-DIFFERENCES'
    init_dict['DERIVATIVES']['eps'] = eps

    # Cleanup
    del init_dict['PROGRAM']['procs']
    del init_dict['SCIPY-BFGS']['eps']
    del init_dict['FORT-BFGS']['eps']
    del init_dict['PRECONDITIONING']

    # Ambiguity was not yet available
    del init_dict['AMBIGUITY']
    pkl.dump(init_dict, open('old/init_dict.respy.pkl', 'wb'))


def run_estimation(which):
    """ Run an estimation with the respective release.
    """
    os.chdir(which)

    from respy import estimate
    from respy import RespyCls

    from respy.python.shared.shared_auxiliary import print_init_dict
    init_dict = pkl.load(open('init_dict.respy.pkl', 'rb'))

    print_init_dict(init_dict)

    respy_obj = RespyCls('test.respy.ini')

    # TODO: There was a bug in version 1.0 which might lead to crit_val not
    # to actually take the lowest value that was visited by the optimizer.
    # So, we reprocess the log file again to be sure.
    estimate(respy_obj)

    crit_val = 1e10
    with open('est.respy.log') as infile:
        for line in infile.readlines():
            list_ = shlex.split(line)
            # Skip empty lines
            if not list_:
                continue
            # Process candidate value
            if list_[0] == 'Criterion':
                try:
                    value = float(list_[1])
                    if value < crit_val:
                        crit_val = value
                except ValueError:
                    pass

    pkl.dump(crit_val, open('crit_val.respy.pkl', 'wb'))

    os.chdir('../')

if __name__ == '__main__':

    if sys.argv[1] == 'prepare':
        version = sys.argv[2]
        install(version)
    elif sys.argv[1] == 'upgrade':
        pip.main(['install', '--upgrade', 'pip'])
    elif sys.argv[1] == 'estimate':
        which = sys.argv[2]
        run_estimation(which)
    else:
        raise NotImplementedError
