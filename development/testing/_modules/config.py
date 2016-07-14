import sys
import os

# Edit PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/testing/_modules', '')
sys.path.insert(0, PACKAGE_DIR)

# Construct path to executables that ensures portability
HOME = os.environ['HOME']
python3_exec = HOME + '/.envs/restudToolbox3/bin/python'
python2_exec = HOME + '/.envs/restudToolbox2/bin/python'

# Directory for specification of baselines
SPEC_DIR = PACKAGE_DIR + '/respy/tests/resources/'
SPECS = ['kw_data_one', 'kw_data_two', 'kw_data_three']