#!/usr/bin/env python
""" This script creates fresh virtual environments for the development of the restudToolbox.
"""

import subprocess
import os

ENV_DIR = os.environ['HOME'] + '/.envs'

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = PROJECT_DIR.replace('/development/_scripts', '')

EXEC_PYTHON_2 = 'python2.7'
EXEC_PYTHON_3 = 'python3.5'

for name in ['restudToolbox', 'restudToolbox2']:

    if name in ['restudToolbox2']:
        exec = '/usr/bin/' + EXEC_PYTHON_2
    else:
        exec = '/usr/bin/' + EXEC_PYTHON_3

    cmd = ['virtualenv', ENV_DIR + '/' + name, '--clear', '--python=' + exec]
    subprocess.check_call(cmd)

    # We also need to add some additional directories to the PYTHONPATH.
    # TODO
    top = ["import sys; sys.__plen = len(sys.path)"]
    bottom = ["import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)"]

    subdirs = []
    subdirs = ['/development/_modules']
    subdirs = ['/respy/tests']
    subdirs = ['/respy/tests/codes']

    dirnames = []
    for subdir in subdirs:
        dirnames += [PROJECT_DIR + subdir]


    path_additions = top

    for dirname in dirnames:
        path_additions += [dirname]

    path_additions += bottom



    # Write out additions.
    fname = ENV_DIR + '/' + name + '/lib/python2.7/site-packages/_virtualenv_path_extensions.pth'

    with open(fname, 'w') as outfile:
        for line in path_additions:
            outfile.write("%s\n" % line)
