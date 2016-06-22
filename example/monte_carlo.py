#!/usr/bin/env python
""" Using the first specification from Keane & Wolpin (1994), we perform
a simple Monte Carlo exercise to ensure the reliability of the impelementation.
"""

# standard library
import shutil
import glob
import os

# project library
import respy

# Let us first simulate a baseline sample and store the results for future reference.
os.mkdir('correct'), os.chdir('correct')
respy_obj = respy.RespyCls('../kw_data_one.ini')
respy.simulate(respy_obj)

respy_obj.unlock()
respy_obj.set_attr('maxfun', 0)
respy_obj.lock()

respy.estimate(respy_obj)


os.chdir('../')

# # TODO: PRINT INIT DICT
# # Now we will estimate a misspecified model on this dataset assuming that agents are myopic.
os.mkdir('static'), os.chdir('static')
respy_obj = respy.RespyCls('../kw_data_one.ini')

respy_obj.unlock()
respy_obj.set_attr('file_est', '../correct/data.respy')
respy_obj.set_attr('delta', 0.00)
# TODO SET
respy_obj.set_attr('maxfun', 100000000)
respy_obj.set_attr('optimizer_used', 'FORT-NEWUOA')
respy_obj.lock()

x, crit_val = respy.estimate(respy_obj)

# Update the respy class instance, with the new parameters.
print('\n\n 1')
print(respy_obj.attr['model_paras'])

respy_obj.update_model_paras(x)
print('\n\n 2')
print(respy_obj.attr['model_paras'])

respy.simulate(respy_obj)
os.chdir('../')

################################################################################
os.mkdir('dynamic'), os.chdir('dynamic')

respy_obj.unlock()
respy_obj.set_attr('delta', 0.95)
respy_obj.lock()

print('\n\n 3')
print(respy_obj.attr['model_paras'])
respy_obj.update_model_paras(x)

print('\n\n 4')
print(respy_obj.attr['model_paras'])

x, crit_val = respy.estimate(respy_obj)

respy_obj.update_model_paras(x)
respy.simulate(respy_obj)
os.chdir('../')