#!/usr/bin/env python
""" This module recomputes some of the key results of Keane & Wolpin (1994).
"""

import respy

# We can simply iterate over the different model specifications outlined in
# Table 1 of the paper.
for spec in ['data_one.ini', 'data_two.ini', 'data_three.ini']:

    # Process relevant model initialization file
    respy_obj = respy.RespyCls(spec)

    # Let us simulate the datasets discussed pn the page 658.
    respy.simulate(respy_obj)

    # To start estimations for the Monte Carlo exercises. For now, we just
    # evaluate the model at the starting values, i.e. maxiter set to zero in
    # the initialization file.
    respy.estimate(respy_obj)
