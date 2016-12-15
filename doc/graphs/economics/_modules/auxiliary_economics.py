import subprocess
def float_to_string(float_):
    """ Get string from a float.
    """
    return '%03.6f' % float_


def cleanup():

    subprocess.check_call(['git', 'clean', '-d', '-f'])