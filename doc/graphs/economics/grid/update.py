#!/usr/bin/env python
""" Updates the results from the production server.
"""

import shutil
import os

REMOTE_DIR = 'acropolis:/home/eisenhauer/structAmbiguity/simulations/exploration/grid'
LOCAL_DIR = '/home/peisenha/structAmbiguity/simulations/exploration'
cmd = 'rsync -arvz ' + REMOTE_DIR + ' ' + LOCAL_DIR

# Cleanup from previous results
for dir_ in [x[0] for x in os.walk('.')]:
    if dir_ in ['.']:
        continue
    shutil.rmtree(dir_)

os.system(cmd)
