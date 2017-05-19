#!/usr/bin/env python
""" This script creates fresh virtual environments for the development of the restudToolbox.
"""

import subprocess
import glob
import os

ENV_DIR = os.environ['HOME'] + '/.envs'
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = PROJECT_DIR.replace('/development/_scripts', '')
SCRIPTS_DIR = PROJECT_DIR + '/respy/_scripts'
BASE_DIR = os.getcwd()


if __name__ == '__main__':

    for name in ['restudToolbox', 'restudToolbox2']:

        if name in ['restudToolbox2']:
            exec_ = '/usr/bin/python2'
        else:
            exec_ = '/usr/bin/python3'

        cmd = ['virtualenv', ENV_DIR + '/' + name, '--clear', '--python=' + exec_]
        subprocess.check_call(cmd)

        # We also need to add some additional directories to the PYTHONPATH.
        top = "import sys; sys.__plen = len(sys.path)"

        bottom = ''
        bottom += "import sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:];"
        bottom += "p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert ="
        bottom += "p+len(new)"

        # Here we collect the manual additions to the PYTHONPATH of the virutal environment.
        subdirs = []
        subdirs += ['/development/_modules']
        subdirs += ['/respy/tests']
        subdirs += ['/respy/tests/codes']

        dirnames = []
        for subdir in subdirs:
            dirnames += [PROJECT_DIR + subdir]

        path_additions = [top]
        for dirname in dirnames:
            path_additions += [dirname]
        path_additions += [bottom]

        os.chdir(ENV_DIR + '/' + name + '/lib')
        fname = glob.glob('python*')[0] + '/site-packages/_virtualenv_path_extensions.pth'

        with open(fname, 'w') as outfile:
            for line in path_additions:
                outfile.write("%s\n".format(line))

        # We now also create links to all scripts. We first do so for the scripts that are included
        # in the package and then also for some auxiliary scripts.
        os.chdir('../bin')
        for src in glob.glob(SCRIPTS_DIR + '/scripts_*.py'):
            dst = 'respy-' + src.split('_')[-1].replace('.py', '')
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        for src in glob.glob(PROJECT_DIR + '/development/_scripts/scripts_*.py'):
            dst = 'respy-' + src.split('_')[-1].replace('.py', '')
            if os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)

        os.chdir(BASE_DIR)

        # Now we switch into the virtual environment and prepare the installation of packages .
        cmd = [ENV_DIR + '/' + name + '/bin/python', '_venn_setup.py']
        subprocess.check_call(cmd)
