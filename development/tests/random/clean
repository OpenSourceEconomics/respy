#!/usr/bin/env python
''' Script to cleanup after a testing run.
'''

# standard library
import shutil
import glob
import os


files = []

''' Clean main.
'''
files += glob.glob('.waf*')

files += glob.glob('*.grm.*')

files += glob.glob('.pkl*')

files += glob.glob('.txt*')

files += glob.glob('.grm.*')

files += glob.glob('*.pkl')

files += glob.glob('*.txt')

files += glob.glob('*.out')

files += glob.glob('*.ini')

files += glob.glob('*.pyc')

files += glob.glob('.seed')

files += glob.glob('.slave-*')

files += glob.glob('.dat*')

''' Clean modules.
    '''
files += glob.glob('modules/*.out*')

files += glob.glob('modules/*.pyc')

for file_ in files:

    try:

        os.remove(file_)

    except:

        try:

            shutil.rmtree(file_)

        except:

            pass
