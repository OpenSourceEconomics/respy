#!/usr/bin/env python
""" Updates the results from the production server.
"""

import shutil
import os

REMOTE_BASE = 'acropolis:/home/eisenhauer/restudToolbox'
REMOTE_BASE += os.getcwd().split('restudToolbox')[1] + '/rslt'

if os.path.exists('rslt'):
    shutil.rmtree('rslt')

os.system('rsync -arvz ' + REMOTE_BASE + ' ' + os.getcwd())
