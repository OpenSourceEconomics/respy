#!/usr/bin/env python


import subprocess

# path to a python interpreter that runs any python script
# under the virtualenv /path/to/virtualenv/
python3_bin = "/home/peisenha/.envs/restudToolbox/bin/python"
python2_bin = "/home/peisenha/.envs/restudToolbox2/bin/python"

# path to the script that must run under the virtualenv
script_file = 'check_vault.py'

# doing execfile() on this file will alter the current interpreter's
# environment so you can import libraries in the virtualenv
activate_this_file = "/path/to/virtualenv/bin/activate_this.py"

execfile(activate_this_file, dict(__file__=activate_this_file))
for python in [python2_bin]:
    p = subprocess.Popen([python, script_file])
    p.wait()