#!/usr/bin/env python
""" This script provisions the required environment for the robustToolbox.
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
            'libevent-dev', 'python3-matplotlib', 'python3-numpy',
            'python3-scipy', 'python3-pandas', 'libfreetype6-dev', 'libxft-dev']

for package in packages:

    os.system('apt-get install -y ' + package)

# Download development environment
os.chdir(GUEST)

if not os.path.exists('robustToolbox'):
    
    os.mkdir('robustToolbox'), os.chdir('robustToolbox')

    os.system('git clone https://github.com/robustToolbox/package.git')


    os.mkdir('documentation'), os.chdir('documentation')

    os.system('git clone https://github.com/robustToolbox/documentation.git')

    os.system('git clone https://github.com/robustToolbox/robustToolbox.github.io.git')


    os.chdir('../../')


shutil.copyfile(HOST + '/bin/pull.py', GUEST + '/robustToolbox/pull')

os.system('chmod 755 ' + GUEST + '/robustToolbox/pull')

# Create and prepare virtual environment. This is still required
# as the toolbox is written in Python 3.
os.system('pip install virtualenv virtualenvwrapper')

if 'virtualenvwrapper' not in open(GUEST + '/.profile').read():
    
    with open(GUEST + '/.profile', 'a+') as file_:

        file_.write('\n' + 'export WORKON_HOME=' + GUEST + '/.envs')

        file_.write('\n' + 'source /usr/local/bin/virtualenvwrapper.sh')

# Initialize virtual environment for development.
os.system('sudo /vagrant/bin/initialize_envs.sh')

# Integration of robustToolbox.
if 'ROBUPY' not in open(GUEST + '/.profile').read():

    with open(GUEST + '/.profile', 'a+') as file_:

        file_.write('\n' + 'export ROBUPY=$HOME/robustToolbox/package')

# Cleanup permissions.
for path in [HOST, GUEST]:

    os.system('chown -R vagrant:vagrant ' + path)
