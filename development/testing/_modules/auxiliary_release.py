import pickle as pkl
import sys
import pip


def install(version):

    print(pip.__version__)
    cmd = ['install', '--no-binary', 'respy', '--extra-index-url']
    cmd += ['https://testpypi.python.org/pypi', 'respy==' + version]
    pip.main(cmd)

    cmd = ['install', 'pytest']
    pip.main(cmd)


def run_estimation(which):
    """ Run an estimation with the respective release.
    """
    import numpy as np

    from respy import estimate
    from respy import RespyCls

    from respy.python.shared.shared_auxiliary import print_init_dict
    init_dict = pkl.load(open('release_' + which + '.respy.pkl', 'rb'))

    print_init_dict(init_dict)

    respy_obj = RespyCls('test.respy.ini')
    crit_val = estimate(respy_obj)[1]
    np.savetxt('.crit_val', np.array(crit_val, ndmin=2))

    import shutil
    shutil.copy('test.respy.ini', 'test.respy.' + which)

if __name__ == '__main__':

    print(sys.argv)
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
