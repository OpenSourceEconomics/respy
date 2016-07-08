#!/usr/bin/env python
""" I will now try to run some estimations.
"""
import time
import sys
import os

sys.path.insert(0, '../../')
from respy import simulate, RespyCls, estimate


def write_information(procs, start, end):
    with open('scalability.respy.log', 'a') as outfile:
        fmt = ' {:>5}    {:>15}\n'
        outfile.write(fmt.format(*[procs, str(int(end - start))]))

# Recompile package without DEBUG flag for optimization.
if True:
    cwd = os.getcwd()
    os.chdir('../../respy')
    assert os.system('./waf distclean; ./waf configure build') == 0
    os.chdir(cwd)

with open('scalability.respy.log', 'w') as outfile:
    outfile.write('\n Scalability of the RESPY package\n')
    outfile.write(' ' + '-' * 32 + '\n\n')
    fmt = ' {:>5}    {:>15}\n\n'
    outfile.write(fmt.format(*['Cores', 'Seconds']))

# Simulate a dataset for scalability exericse.
print('\n Starting simulation of sample ...')
respy_obj = RespyCls('model.respy.ini')
simulate(respy_obj)
print(' ... finished')

# Run estimation with varying number of cores. All other elements of the
# initialization file remain the same. The baseline is always provided by the
# scalar version
print('\n\n Starting scalar estimation ...')
respy_obj.attr['is_parallel'] = False
start = time.time()
estimate(respy_obj)
end = time.time()
write_information(0, start, end)
print(' ... finished')

print('\n\n Starting parallel estimations ...')
for proc in [2, 3]:
    respy_obj.attr['is_parallel'] = True
    respy_obj.attr['num_procs'] = proc

    start = time.time()
    estimate(respy_obj)
    end = time.time()
    write_information(proc, start, end)
print(' ... finished')





