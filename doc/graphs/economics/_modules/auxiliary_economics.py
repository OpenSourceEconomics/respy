import subprocess
import shutil
import glob
import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
INIT_FILE = BASE_DIR.replace('/economics/_modules', '') + '/graphs.respy.ini'
GRID_RSLT = BASE_DIR.replace('/_modules', '') + '/grid_ambiguity/rslt'


def move_subdirectory():
    """ Create a clean directory for the results and enter.
    """
    if os.path.exists('rslt'):
        shutil.rmtree('rslt')
    os.mkdir('rslt')

    os.chdir('rslt')


def float_to_string(float_):
    """ Get string from a float.
    """
    return '%03.6f' % float_


def cleanup():
    """ Cleanup previous results.
    """
    subprocess.check_call(['git', 'clean', '-d', '-f'])


def get_float_directories(dirname=None):
    """ Get directories that have a float-type name.
    """
    cwd = os.getcwd()
    if dirname is not None:
        os.chdir(dirname)

    candidates = glob.glob('*')
    directories = []
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        try:
            float(candidate)
        except ValueError:
            continue
        directories += [float(candidate)]
    directories.sort()

    os.chdir(cwd)

    return directories
