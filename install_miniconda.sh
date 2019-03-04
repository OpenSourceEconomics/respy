set -e

if [[ "$TRAVIS_PYTHON_VERSION" == "3.7-dev" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O miniconda.sh;
fi
bash miniconda.sh -b -p /home/travis/miniconda
export PATH=/home/travis/miniconda/bin:$PATH
conda config --set always_yes yes --set changeps1 no
conda info -a
