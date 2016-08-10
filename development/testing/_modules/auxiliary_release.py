import pip
import sys


def install(version):

    pip.main(['install', '--no-binary', 'respy', 'respy==' + version])

    # Run full unit test suite to test installation.
    pip.main(['install', 'pytest'])


def run_estimation():
    """ Run an estimation with the respective release.
    """
    import numpy as np

    from respy import estimate
    from respy import RespyCls

    respy_obj = RespyCls('test.respy.ini')
    crit_val = estimate(respy_obj)[1]
    np.savetxt('.crit_val', np.array(crit_val, ndmin=2))


if __name__ == '__main__':

    if len(sys.argv) > 1:
        version = sys.argv[1]
        install(version)

    else:
        run_estimation()
