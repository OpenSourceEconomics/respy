apt get update
subprocess.check_call(['apt-get', 'upgrade', '-y'])

# installed version is too old.
apt_packages = ['python-pip', 'gfortran', 'libatlas-dev', 'libatlas-base-dev']
apt_packages += ['git', 'python-matplotlib', 'python-numpy', 'python-scipy']
pip install virtualenv
git clone https://github.com/restudToolbox/package.git

Some problems updating pip ....


subprocess.call(['pip', 'install', '--upgrade', 'pip>=8.1'])

pip install --no-binary respy respy

sudo apt-get install libcr-dev mpich2 mpich2-doc