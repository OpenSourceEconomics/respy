#!/usr/bin/env python


import subprocess

# path to a python interpreter that runs any python script
# under the virtualenv /path/to/virtualenv/
python3_bin = "/home/peisenha/.envs/restudToolbox/bin/python"
python2_bin = "/home/peisenha/.envs/restudToolbox2/bin/python"

# path to the script that must run under the virtualenv
script_file = 'check_vault.py'

for python_bin in [python2_bin, python3_bin]:
    assert (subprocess.call([python_bin, script_file]) == 0)
