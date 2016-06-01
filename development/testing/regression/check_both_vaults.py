#!/usr/bin/env python


import subprocess

# path to a python interpreter that runs any python script
# under the virtualenv /path/to/virtualenv/
python3_bin = "/home/peisenha/.envs/restudToolbox/bin/python"
python2_bin = "/home/peisenha/.envs/restudToolbox2/bin/python"

# path to the script that must run under the virtualenv
script_file = 'run_single_vault.py'

for python in [python3_bin, python2_bin]:
    subprocess.Popen([python, script_file])