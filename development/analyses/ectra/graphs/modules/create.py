#!/usr/bin/env python
""" This module creates some graphs for the economy in the case of ambiguity.
"""
# standard library
import shutil
import os

# project library
from auxiliary import track_schooling_over_time
from auxiliary import get_ambiguity_levels
from auxiliary import track_final_choices

from auxiliary import plot_schooling_ambiguity
from auxiliary import plot_choices_ambiguity

# module-wide variables
LABELS_SUBSET = ['0.000', '0.010', '0.020']
MAX_PERIOD = 25
HOME = os.environ['ROBUPY'] + '/development/analyses/ectra/graphs'

# Preparations, starting with a clean slate.
if os.path.exists('rslts'):
    shutil.rmtree('rslts')
os.makedirs('rslts')

# Here we investigate the final choice distribution over time. We iterate
# through all results and store the final choice distribution.
levels = get_ambiguity_levels()
shares_ambiguity = track_final_choices(levels)
plot_choices_ambiguity(levels, shares_ambiguity)

# Here we investigate the evolution of schooling over time for selected
# levels of ambiguity. As schooling tends to zero against the end of the
# life-cycle, the graph is cut off after a couple of periods.
shares_time = track_schooling_over_time(levels)
plot_schooling_ambiguity(LABELS_SUBSET, MAX_PERIOD, shares_time)