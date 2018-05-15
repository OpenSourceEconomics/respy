import os

# Edit PYTHONPATH
PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))
PACKAGE_DIR = PACKAGE_DIR.replace('development/modules', '')

# Directory for specification of baselines
SPEC_DIR = PACKAGE_DIR + '/respy/tests/resources/'
SPECS = ['kw_data_one', 'kw_data_two', 'kw_data_three']