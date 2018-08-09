import pickle as pkl

import subprocess
import shutil
import shlex
import glob
import json
import copy
import sys
import pip
import os


def install(version):
    """ Prepare the
    """
    if version == 'current':
        cwd = os.getcwd()
        os.chdir('../../../')
        # We need to make sure that all required packages are available inside the
        # virtual environment first. Then we can install the package itself in developer mode.
        pip.main(['install', '-r', 'requirements.txt'])
        subprocess.check_call([sys.executable, 'setup.py', 'develop'])
        os.chdir(cwd)
    else:
        cmd = ['install', '-vvv', '--no-binary', 'respy', 'respy==' + version]
        pip.main(cmd)

        # PYTEST is part of the package requirements for later releases.
        if version == '1.0.0':
            cmd = ['install', 'pytest']
            pip.main(cmd)


def prepare_release_tests(constr, OLD_RELEASE, NEW_RELEASE):
    """ This function ensures that the right preparations are applied to the initialization files.
    """
    if OLD_RELEASE == '1.0.0' and NEW_RELEASE in ['2.0.0.dev7', '2.0.0.dev8']:
        prepare_release_tests_1(constr)
    elif OLD_RELEASE == '2.0.0.dev7' and NEW_RELEASE == '2.0.0.dev8':
        prepare_release_tests_2(constr)
    elif OLD_RELEASE == '2.0.0.dev8' and NEW_RELEASE == '2.0.0.dev9':
        prepare_release_tests_3(constr)
    elif OLD_RELEASE == '2.0.0.dev9' and NEW_RELEASE == '2.0.0.dev10':
        prepare_release_tests_3(constr)
    elif OLD_RELEASE == '2.0.0.dev10' and NEW_RELEASE == '2.0.0.dev11':
        prepare_release_tests_4(constr)
    elif OLD_RELEASE == '2.0.0.dev11' and NEW_RELEASE == '2.0.0.dev12':
        prepare_release_tests_5(constr)
    elif OLD_RELEASE == '2.0.0.dev12' and NEW_RELEASE == '2.0.0.dev14':
        # In principle it is better to simulate an observed dataset. However, it was changed between
        # the two releases v2.0.0.dev12 and v2.0.0.dev14. So we rely on a synthetic simulation in
        # these cases.
        open('.simulate_observed.cfg', 'w').close()
        prepare_release_tests_6(constr)
    elif OLD_RELEASE == '2.0.0.dev14' and NEW_RELEASE == '2.0.0.dev15':
        no_preparations_required(constr)
    elif OLD_RELEASE == '2.0.0.dev15' and NEW_RELEASE == '2.0.0.dev16':
        prepare_release_tests_7(constr)
    elif OLD_RELEASE == '2.0.0.dev16' and NEW_RELEASE == '2.0.0.dev17':
        prepare_release_tests_8(constr)
    elif OLD_RELEASE == '2.0.0.dev17' and NEW_RELEASE == '2.0.0.dev18':
        prepare_release_tests_9(constr)
    elif OLD_RELEASE == '2.0.0.dev18' and NEW_RELEASE == '2.0.0.dev19':
        prepare_release_tests_10(constr)
    elif OLD_RELEASE == '2.0.0.dev19' and NEW_RELEASE == '2.0.0.dev20':
        no_preparations_required(constr)
    else:
        raise AssertionError('Misspecified request ...')


def prepare_release_tests_1(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    # This script is also imported (but not used) for the creation of the virtual environments.
    # Thus, the imports might not be valid when starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['level'] = 0.00
    constr['fixed_ambiguity'] = True
    constr['fixed_delta'] = True
    constr['file_est'] = '../data.respy.dat'

    init_dict = generate_init(constr)

    # In the old release, there was just one location to define the step size for all derivative
    # approximations.
    eps = np.round(init_dict['SCIPY-BFGS']['eps'], decimals=15)
    init_dict['PRECONDITIONING']['eps'] = eps
    init_dict['FORT-BFGS']['eps'] = eps

    # We also endogenized the discount rate, so we need to restrict the analysis to estimations
    # where the discount rate is fixed.
    init_dict['BASICS']['delta'] = init_dict['BASICS']['coeffs'][0]

    # We did not have any preconditioning implemented in the PYTHON version initially. We had to
    # switch the preconditioning scheme in the new release and now use the absolute value and
    # thus preserve the sign of the derivative.
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

    # The concept of bounds for parameters was not available and the coefficients in the
    # initialization file were only printed to the first four digits.
    for label in ['HOME', 'OCCUPATION A', 'OCCUPATION B', 'EDUCATION', 'SHOCKS']:
        num = len(init_dict[label]['fixed'])
        coeffs = np.round(init_dict[label]['coeffs'], decimals=4).tolist()
        init_dict[label]['bounds'] = [(None, None)] * num
        init_dict[label]['coeffs'] = coeffs

    # In the original release we treated TAU as an integer when printing to file by accident.
    init_dict['ESTIMATION']['tau'] = int(init_dict['ESTIMATION']['tau'])
    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))

    # Added more fine grained scaling. Needs to be aligned across old/new with identity or flag
    # False first and then we want to allow for more nuanced check.
    init_dict['SCALING'] = dict()
    init_dict['SCALING']['flag'] = (init_dict['PRECONDITIONING']['type'] == 'gradient')
    init_dict['SCALING']['minimum'] = init_dict['PRECONDITIONING']['minimum']

    # More flexible parallelism. We removed the extra section onn parallelism.
    init_dict['PARALLELISM'] = dict()
    init_dict['PARALLELISM']['flag'] = init_dict['PROGRAM']['procs'] > 1
    init_dict['PARALLELISM']['procs'] = init_dict['PROGRAM']['procs']

    # We had a section that enforced the same step size for the derivative calculation in each.
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

    json.dump(init_dict, open('old/init_dict.respy.json', 'w'))


def prepare_release_tests_2(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['level'] = 0.00
    constr['fixed_ambiguity'] = True
    constr['file_est'] = '../data.respy.dat'

    init_dict = generate_init(constr)

    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))

    # In the old version, we did not allow for variability in the standard deviations.
    del init_dict['AMBIGUITY']['mean']

    json.dump(init_dict, open('old/init_dict.respy.json', 'w'))


def prepare_release_tests_3(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['precond_type'] = np.random.choice(['identity', 'gradient'])
    init_dict = generate_init(constr)

    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))
    json.dump(init_dict, open('old/init_dict.respy.json', 'w'))


def prepare_release_tests_4(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    init_dict = generate_init(constr)

    # We need to make sure that there are no effects on the reentry costs, as there are
    # separately estimation in the new release. They are fixed during an estimation there as well.
    init_dict['EDUCATION']['fixed'][-1] = True
    json.dump(init_dict, open('old/init_dict.respy.json', 'w'))

    # We added sheepskin effects to the wage equations.
    init_dict['OCCUPATION A']['coeffs'] += [0.0, 0.0]
    init_dict['OCCUPATION A']['fixed'] += [True, True]
    init_dict['OCCUPATION A']['bounds'] += [[None, None], [None, None]]

    init_dict['OCCUPATION B']['coeffs'] += [0.0, 0.0]
    init_dict['OCCUPATION B']['fixed'] += [True, True]
    init_dict['OCCUPATION B']['bounds'] += [[None, None], [None, None]]

    # We are also splitting up the re-entry costs between high school and college graduation
    init_dict['EDUCATION']['coeffs'].append(init_dict['EDUCATION']['coeffs'][-1])
    init_dict['EDUCATION']['fixed'].append(True)
    init_dict['EDUCATION']['bounds'].append(init_dict['EDUCATION']['bounds'][-1])

    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))


def prepare_release_tests_5(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    init_dict = generate_init(constr)

    json.dump(init_dict, open('old/init_dict.respy.json', 'w'))

    # We added an additional coefficient indicating whether there is any experience in a
    # particular job.
    init_dict['OCCUPATION A']['coeffs'].append(0.00)
    init_dict['OCCUPATION A']['bounds'].append([None, None])
    init_dict['OCCUPATION A']['fixed'].append(True)

    init_dict['OCCUPATION B']['coeffs'].append(0.00)
    init_dict['OCCUPATION B']['bounds'].append([None, None])
    init_dict['OCCUPATION B']['fixed'].append(True)

    # This release rescaled the squared term in the experience variable by 100. The presence of
    # the scratch file ensures that this is undone
    open('new/.restud.respy.scratch', 'w').close()

    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))


def prepare_release_tests_6(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    # This script is also imported (but not used) for the creation of the virtual environments.
    # Thus, the imports might not be valid when starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    # Unfortunately, we needed to perform edits to the likelihood function which breaks
    # comparability in all cases but for a model with a single period. We also changed the
    # treatment of inadmissible states, so we need to ensure that these are not relevant.
    constr['periods'] = 1
    edu_start = np.random.randint(1, 5)
    constr['edu'] = (edu_start, edu_start + 100)

    init_dict = generate_init(constr)

    old_dict = init_dict.copy()
    old_dict['EDUCATION']['start'] = old_dict['EDUCATION']['start'][0]
    del old_dict['TYPE_SHARES'], old_dict['TYPE_SHIFTS'], old_dict['EDUCATION']['share']
    json.dump(old_dict, open('old/init_dict.respy.json', 'w'))

    # We need to specify a sample with a baseline type only a single initial condition.
    init_dict['TYPE_SHIFTS'] = dict()
    init_dict['TYPE_SHIFTS']['coeffs'] = [0.0, 0.0, 0.0, 0.0]
    init_dict['TYPE_SHIFTS']['bounds'] = [(None, None), (None, None), (None, None), (None, None)]
    init_dict['TYPE_SHIFTS']['fixed'] = [True, True, True, True]

    init_dict['TYPE_SHARES'] = dict()
    init_dict['TYPE_SHARES']['coeffs'] = [1.0]
    init_dict['TYPE_SHARES']['bounds'] = [(0.0, None)]
    init_dict['TYPE_SHARES']['fixed'] = [True]

    init_dict['EDUCATION']['start'] = [init_dict['EDUCATION']['start']]
    init_dict['EDUCATION']['share'] = [1.0]
    init_dict['EDUCATION']['max'] = init_dict['EDUCATION']['max']

    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))


def prepare_release_tests_7(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    # This script is also imported (but not used) for the creation of the virtual environments.
    # Thus, the imports might not be valid when starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    # This dictionary is generated by the current code, so we first store the new initialization
    # dictionary.
    init_dict = generate_init(constr)

    new_dict = init_dict.copy()
    for name in ['OCCUPATION A', 'OCCUPATION B', 'HOME']:
        new_dict[name]['coeffs'][-2:] = [0.0, 0.0]
        new_dict[name]['bounds'][-2:] = [[None, None], [None, None]]
        new_dict[name]['fixed'][-2:] = [True, True]

    new_dict['EDUCATION']['coeffs'][2] = 0.0
    new_dict['EDUCATION']['bounds'][2] = [None, None]
    new_dict['EDUCATION']['fixed'][2] = True

    new_dict['EDUCATION']['coeffs'][-2:] = [0.0, 0.0]
    new_dict['EDUCATION']['bounds'][-2:] = [[None, None], [None, None]]
    new_dict['EDUCATION']['fixed'][-2:] = [True, True]
    json.dump(new_dict, open('new/init_dict.respy.json', 'w'))

    old_dict = init_dict.copy()
    for name in ['OCCUPATION A', 'OCCUPATION B', 'HOME']:
        for label in ['coeffs', 'bounds', 'fixed']:
            old_dict[name][label] = old_dict[name][label][:-2]
    for label in ['coeffs', 'bounds', 'fixed']:
        del old_dict['EDUCATION'][label][2]
        del old_dict['EDUCATION'][label][-2:]
    json.dump(old_dict, open('old/init_dict.respy.json', 'w'))


def prepare_release_tests_8(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    # This script is also imported (but not used) for the creation of the virtual environments.
    # Thus, the imports might not be valid when starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    # As we added more information about lagged activities the interpolation estimation does not
    # yield the same results.
    constr['flag_interpolation'] = False
    init_dict = generate_init(constr)

    new_dict = init_dict.copy()
    for name in ['OCCUPATION A', 'OCCUPATION B']:
        new_dict[name]['coeffs'][8] = 0.00
        new_dict[name]['bounds'][8] = [None, None]
        new_dict[name]['fixed'][8] = True
    json.dump(new_dict, open('new/init_dict.respy.json', 'w'))

    # We need to remove the coefficient denoting whether an individual worked in the same
    # occupation last period.
    old_dict = init_dict.copy()
    for name in ['OCCUPATION A', 'OCCUPATION B']:
        for label in ['coeffs', 'bounds', 'fixed']:
            old_dict[name][label].pop(8)
    json.dump(old_dict, open('old/init_dict.respy.json', 'w'))


def prepare_release_tests_9(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    def transform_to_logit(shares):
        """ This function transform
        """
        num_types = len(shares)
        denominator = (1.0 / shares[0])

        coeffs = []
        for i in range(num_types):
            coeffs += [np.log(shares[i] * denominator)]

        return coeffs

    # This script is also imported (but not used) for the creation of the virtual environments.
    # Thus, the imports might not be valid when starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.auxiliary import get_valid_shares
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['flag_estimation'] = True
    init_dict = generate_init(constr)

    # I now need to determine the number of types and sample the fixed type probabilites.
    num_types = int((len(init_dict['TYPE SHARES']['coeffs']) / 2) + 1)
    shares = get_valid_shares(num_types)

    new_dict = init_dict.copy()
    coeffs = transform_to_logit(shares)

    new_dict['TYPE SHARES']['coeffs'] = []
    new_dict['TYPE SHARES']['fixed'] = []
    new_dict['TYPE SHARES']['bounds'] = []

    for coeff in coeffs[1:]:
        new_dict['TYPE SHARES']['coeffs'] += [coeff]
        new_dict['TYPE SHARES']['coeffs'] += [0.0]
        new_dict['TYPE SHARES']['fixed'] += [True, True]
        new_dict['TYPE SHARES']['bounds'] += [[None, None], [None, None]]

    json.dump(new_dict, open('new/init_dict.respy.json', 'w'))

    old_dict = init_dict.copy()

    old_dict['TYPE_SHIFTS'] = old_dict['TYPE SHIFTS']
    del old_dict['TYPE SHIFTS']

    old_dict['TYPE_SHARES'] = dict()
    old_dict['TYPE_SHARES']['coeffs'] = shares
    old_dict['TYPE_SHARES']['fixed'] = [True] * num_types
    old_dict['TYPE_SHARES']['bounds'] = [[0.00, None]] * num_types

    json.dump(old_dict, open('old/init_dict.respy.json', 'w'))


def prepare_release_tests_10(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """

    # This script is also imported (but not used) for the creation of the virtual environments.
    # Thus, the imports might not be valid when starting with a clean slate.
    import numpy as np

    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    constr['flag_estimation'] = True
    init_dict = generate_init(constr)

    # We aligned the indicator functions with the KW1997 setup and also added a constant term for
    # the general rewards. Finally, we added the common rewards.
    for label in ['OCCUPATION A', 'OCCUPATION B']:
        for j in range(8, 15):
            init_dict[label]['coeffs'][j] = 0
            init_dict[label]['bounds'][j] = (None, None)
            init_dict[label]['fixed'][j] = True

    for j in [0, 1]:
        init_dict['COMMON']['coeffs'][j] = 0
        init_dict['COMMON']['bounds'][j] = (None, None)
        init_dict['COMMON']['fixed'][j] = True

    new_dict = copy.deepcopy(init_dict)

    # We swapped to the order to align it with the KW1997 setup.
    for label in ['OCCUPATION A', 'OCCUPATION B']:
        new_dict[label]['coeffs'][8], new_dict[label]['coeffs'][11] = \
            new_dict[label]['coeffs'][11], new_dict[label]['coeffs'][8]
        new_dict[label]['coeffs'][9], new_dict[label]['coeffs'][12] = \
            new_dict[label]['coeffs'][12], new_dict[label]['coeffs'][9]

    json.dump(new_dict, open('new/init_dict.respy.json', 'w'))

    old_dict = copy.deepcopy(init_dict)
    for label in ['OCCUPATION A', 'OCCUPATION B']:
        for name in ['coeffs', 'bounds', 'fixed']:
            old_dict[label][name].pop(10)

    del old_dict['COMMON']

    json.dump(old_dict, open('old/init_dict.respy.json', 'w'))


def no_preparations_required(constr):
    """ This function prepares the initialization files so that they can be processed by both
    releases under investigation. The idea is to have all hand-crafted modifications grouped in
    this function only.
    """
    sys.path.insert(0, '../../../respy/tests')
    from respy.tests.codes.random_init import generate_init

    # Prepare fresh subdirectories
    for which in ['old', 'new']:
        if os.path.exists(which):
            shutil.rmtree(which)
        os.mkdir(which)

    init_dict = generate_init(constr)

    json.dump(init_dict, open('new/init_dict.respy.json', 'w'))
    json.dump(init_dict, open('old/init_dict.respy.json', 'w'))


def run_estimation(which):
    """ Run an estimation with the respective release.
    """
    os.chdir(which)

    import numpy as np

    from respy import estimate
    from respy import RespyCls

    from respy.pre_processing.model_processing import write_init_file

    # We need to make sure that the function simulate_observed() is imported from the original
    # package. Otherwise dependencies might not work properly.
    import respy
    sys.path.insert(0, os.path.dirname(respy.__file__) + '/tests')
    from respy.tests.codes.auxiliary import simulate_observed

    init_dict = json.load(open('init_dict.respy.json', 'r'))

    # There was a change in the setup for releases after 1.00. This is only required when
    # comparing to v1.0.0.
    if '1.0.0' in sys.executable:
        init_dict['SHOCKS']['fixed'] = np.array(init_dict['SHOCKS']['fixed'])

    write_init_file(init_dict)

    respy_obj = RespyCls('test.respy.ini')

    # This flag ensures a clean switch to the synthetic simulation for cases where the
    # simulate_observed() was changed in between releases.
    if os.path.exists('../.simulate_observed.cfg'):
        respy.simulate(respy_obj)
    else:
        simulate_observed(respy_obj)

    # This flag ensures that the change in the truncation of the wage variable has no effect. We
    # simply copy the dataset from the new release to the old.
    if ('2.0.0.dev20' in sys.executable) and ('/new' in os.getcwd()):
        fnames = glob.glob('data.respy.*')
        for fname in fnames:
            shutil.copy('../old/' + fname, '.')

    # Moving from 2.0.0.dev17 to 2.0.0.dev18 breaks the equality because the simulated datasets
    # differ. So, we just copy the one from old. However, this is only relevant if 2.0.0.dev18 is
    # the candidate.
    if ('2.0.0.dev18' in sys.executable) and ('/new' in os.getcwd()):
        os.chdir('../old')
        files = glob.glob('data.respy.*')
        for file in files:
            shutil.copy(file, '../new')
        os.chdir('../new')

    _, crit_val = estimate(respy_obj)

    # There was a bug in version 1.0 which might lead to crit_val not to actually take the lowest
    # value that was visited by the optimizer. So, we reprocess the log file again to be sure.
    if '1.0.0' in sys.executable:
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
