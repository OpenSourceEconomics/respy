#!/usr/bin/env python

DEBUG_OPTIONS = ['-O2', '-Wall', '-Wline-truncation', '-Wcharacter-truncation']
DEBUG_OPTIONS += ['-Wsurprising', '-Waliasing', '-Wimplicit-interface']
DEBUG_OPTIONS += ['-Wunused-parameter', '-fwhole-file', '-fcheck=all']
DEBUG_OPTIONS += ['-fbacktrace', '-g', '-fmax-errors=1', '-ffpe-trap=invalid']

#DEBUG_OPTIONS = []
import os
import sys
import numpy as np

#
# # WHAT WHEN number of procs larger than states, does zero work.
# index_set = range(20)
# num_procs = 4
#
# disply = np.zeros(4)
#
#
# total = len(index_set)
#
#
# print(disply)
#
# j = 0
# for i in range(total):
#
#     if j == num_procs:
#         j = 0
#
#     disply[j] += 1
#
#     j = j + 1
#
#
# #print(disply)
#
#
# #sys.exit('prototyping')
#
os.system('git clean -d -f')

lib_dir = '/home/peisenha/restudToolbox/package/respy/.bld/fortran'
inc_dir = '/home/peisenha/restudToolbox/package/respy/.bld'

#
for fname in ['master', 'slave']:

    cmd = 'mpif90 ' + fname + '.f90 ' + '-o ' + fname + ' '  \
          + ' '.join(DEBUG_OPTIONS) + ' -I' + inc_dir + ' -L' + lib_dir + ' ' \
                                                                        '-lresfort ' \
                                      '-llapack '
    print(cmd, '\n')

    assert os.system(cmd) == 0

os.system('mpiexec ./master')


# Compile testing file.
#cmd = 'mpif90 testing_scalar_parallel.f90 ' + '-o testing  '  \
#          + ' '.join(DEBUG_OPTIONS) + ' -Iinclude/ -Llib/ -lresfort -llapack '
#print(cmd, '\n')

#assert os.system(cmd) == 0









