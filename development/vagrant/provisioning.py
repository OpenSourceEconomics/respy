#!/usr/bin/env python
""" This script provisions the required development environment for the
    robustToolbox.
"""

# standard library
import shutil
import os

# Module-wide variables
GUEST, HOST = '/home/vagrant', '/vagrant'

''' Provisioning
'''
os.system('apt-get update -y')


# Additional system-wide software.
packages = ['build-essential', 'gfortran', 'python3-pip', 'python-pip', 'git',
            'libblas-dev', 'libatlas-base-dev', 'liblapack-dev',
            'libyaml-cpp-dev', 'cython3', 'python-dev', 'python3-dev',
            'libevent-dev']

for package in packages:

    os.system('apt-get install -y ' + package)

# Download development environment
os.chdir(GUEST)

if not os.path.exists('robustToolbox'):
    
    os.mkdir('robustToolbox')

    os.chdir('robustToolbox')

    os.system('git clone https://github.com/robustToolbox/package.git')

    os.system('git clone https://github.com/robustToolbox/development.git')

    os.chdir('../')

shutil.copyfile(HOST + '/pull.py', GUEST + '/robustToolbox/pull')

os.system('chmod 755 ' + GUEST + '/robustToolbox/pull')

# Create and prepare virtual environment. This is still required
# as I am working with Python3.
os.system('pip install virtualenv virtualenvwrapper')

if 'virtualenvwrapper' not in open(GUEST + '/.profile').read():
    
    with open(GUEST + '/.profile', 'a+') as file_:

        file_.write('\n' + 'export WORKON_HOME=' + GUEST + '/.envs')

        file_.write('\n' + 'source /usr/local/bin/virtualenvwrapper.sh')

# Initialize virtual environment for development.
#
#   An issue arises, when the order of package installation does not honor
#   potential dependencies.
#
os.system('sudo /vagrant/initialize_envs.sh')

# Integration of robustToolbox.
if 'ROBUPY' not in open(GUEST + '/.profile').read():

    with open(GUEST + '/.profile', 'a+') as file_:

        file_.write('\n' + 'export ROBUPY=$HOME/robustToolbox/package')

# Cleanup permissions.
for path in [HOST, GUEST]:

    os.system('chown -R vagrant:vagrant ' + path)
