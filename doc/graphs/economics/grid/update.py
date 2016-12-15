#!/usr/bin/env python
""" Updates the results from the production server.
"""

import shutil
import os

REMOTE_DIR = 'acropolis:/home/eisenhauer/restudToolbox/package/doc/graphs/economics/grid/rslt'
LOCAL_DIR = '/home/peisenha/restudToolbox/package/doc/graphs/economics/grid'
cmd = 'rsync -arvz ' + REMOTE_DIR + ' ' + LOCAL_DIR

# Cleanup results from a previous run.
if os.path.exists('rslt'):
    shutil.rmtree('rslt')

os.system(cmd)
