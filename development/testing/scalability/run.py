#!/usr/bin/env python
""" We perform a simple scalability exercise to ensure the reliability of the
RESPY package.
"""
import argparse
import glob
import sys

sys.path.insert(0, '../_modules')
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import compile_package
from auxiliary_scalability import run
from auxiliary_shared import cleanup
from config import SPECS

import shlex


def get_durations():

    rslt = dict()
    labels = []
    grid_slaves = []

    with open('scalability.respy.info', 'r') as infile:

        for line in infile.readlines():

            # Split line
            list_ = shlex.split(line)

            if not list_:
                continue

            if list_[0] == 'Slaves':
                continue

            # Create key for each of the data specifications.
            if 'kw_data' in list_[0]:
                label = list_[0]
                rslt[label] = dict()

                labels += [label]

            # Process the interesting lines.
            if len(list_) == 6:


                from time import mktime
                from datetime import datetime
                from datetime import datetime, timedelta

                num_slaves = int(list_[0])
                grid_slaves += [num_slaves]

                t = datetime.strptime(list_[5],"%H:%M:%S")
                duration = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)


                rslt[label][num_slaves] = duration

    # Remove all duplicates from the grid of slaves.
    grid_slaves = sorted(list(set(grid_slaves)))

    return rslt, labels, grid_slaves

import matplotlib.pyplot as plt
import matplotlib
import datetime

def timeTicks(x, pos):

    d = datetime.timedelta(seconds=x)
    return str(d)

def check_scalability(args):


    if args.is_finalize:

        # TODO: aGGREGATE IN FUNCTION, CALL IN THE END TOW AND SEND WITH EMAIL
        rslt, labels, grid_slaves = get_durations()

        print(rslt, labels, grid_slaves)

        ax = plt.figure(figsize=(12, 8)).add_subplot(111)
        ys = []
        for num_slaves in grid_slaves:
            ys += [rslt[labels[0]][num_slaves].total_seconds()]

        print(ys)
        xs = grid_slaves
        formatter = matplotlib.ticker.FuncFormatter(timeTicks)

        ax.plot(xs, ys)
        ax.yaxis.set_major_formatter(formatter)

        ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
            right='off')

        # y axis
        ax.set_ylim([0.0, 5500])
        ax.yaxis.get_major_ticks()[0].set_visible(False)
        ax.set_ylabel('Hours', fontsize=16)

        # x axis
        ax.set_xlabel('Number of Slaves', fontsize=16)
        ax.set_xticks(grid_slaves)

        # # Write out to
        plt.savefig('scalability.respy.png', bbox_inches='tight',
                     format='png')

        return

    cleanup()

    if args.is_compile:
        compile_package()

    ''' Details of the scalability exercise can be specified in the code block
    below. Note that only deviations from the benchmark initialization files
    need to be addressed.
    '''
    spec_dict = dict()
    spec_dict['maxfun'] = 2000
    spec_dict['optimizer_used'] = 'FORT-NEWUOA'

    grid_slaves = [0, 2, 5, 7, 10]

    if args.is_debug:
        grid_slaves = [0, 2]

        spec_dict['maxfun'] = 200
        spec_dict['num_draws_emax'] = 5
        spec_dict['num_draws_prob'] = 3
        spec_dict['num_agents_est'] = 100
        spec_dict['num_agents_sim'] = spec_dict['num_agents_est']

        spec_dict['scaling'] = [False, 0.00001]
        spec_dict['num_periods'] = 3

    for spec in SPECS:
        run(spec_dict, spec + '.ini', grid_slaves)

    aggregate_information('scalability')
    send_notification('scalability')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check reliability',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--debug', action='store_true', dest='is_debug',
        default=False, help='debug specification')

    parser.add_argument('--compile', action='store_true', dest='is_compile',
        default=False, help='compile RESPY package')

    parser.add_argument('--finalize', action='store_true', dest='is_finalize',
        default=False, help='just create graph')


    check_scalability(parser.parse_args())
