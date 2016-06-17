#!/usr/bin/env python
import subprocess

from config import python2_exec
from config import python3_exec

# path to the script that must run under the virtualenv
script_file = 'check_vault.py'

for python_bin in [python2_exec, python3_exec]:
    subprocess.check_call([python_bin, script_file])
