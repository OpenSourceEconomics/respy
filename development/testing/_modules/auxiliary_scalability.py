import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
import shlex
import os

from auxiliary_shared import strfdelta
from config import SPEC_DIR

import respy


def get_durations():
    rslt, labels, grid_slaves = dict(), [], []

    with open('scalability.respy.base', 'r') as infile:
        for line in infile.readlines():
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
                num_slaves = int(list_[0])
                grid_slaves += [num_slaves]
                t = datetime.strptime(list_[5], "%H:%M:%S")
                duration = timedelta(hours=t.hour, minutes=t.minute,
                                     seconds=t.second)
                rslt[label][num_slaves] = duration

    # Remove all duplicates from the grid of slaves.
    grid_slaves = sorted(list(set(grid_slaves)))

    return rslt, labels, grid_slaves


def linear_gains(num_slaves, ys):
    if num_slaves == 0:
        return ys[0]
    else:
        return ys[0] / num_slaves


def plot_scalability():
    rslt, labels, grid_slaves = get_durations()

    ax = plt.figure(figsize=(12, 8)).add_subplot(111)

    ys = []
    for num_slaves in grid_slaves:
        ys += [rslt[labels[0]][num_slaves].total_seconds()]

    formatter = matplotlib.ticker.FuncFormatter(ylabel_formatting)

    ax.plot(grid_slaves, ys, linewidth=5, label='RESPY Package',
            color='red', alpha=0.8)

    func_lin = []
    for num_slaves in grid_slaves:
        func_lin += [linear_gains(num_slaves, ys)]

    ax.plot(grid_slaves, func_lin, linewidth=1, linestyle='--',
            label='Linear Benchmark', color='black')

    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(labelsize=18, direction='out', axis='both', top='off',
                   right='off')

    ax.set_ylim([0.0, 6500])
    ax.yaxis.get_major_ticks()[0].set_visible(False)
    ax.set_ylabel('Hours', fontsize=16)

    ax.set_xlabel('Number of Slaves', fontsize=16)
    ax.set_xticks(grid_slaves)

    # Set up legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            fancybox=False, frameon=False, shadow=False,
            ncol=2, fontsize=20)

    plt.savefig('scalability.respy.png', bbox_inches='tight', format='png')


def ylabel_formatting(x, _):
    d = timedelta(seconds=x)
    return str(d)


def run(spec_dict, fname, grid_slaves):
    dirname = fname.replace('.ini', '')
    os.mkdir(dirname), os.chdir(dirname)

    respy_obj = respy.RespyCls(SPEC_DIR + fname)

    respy_obj.unlock()
    respy_obj.set_attr('file_est', '../data.respy.dat')
    for key_ in spec_dict.keys():
        respy_obj.set_attr(key_, spec_dict[key_])
    respy_obj.lock()

    # Simulate the baseline dataset, which is used regardless of the number
    # of slaves.
    respy.simulate(respy_obj)
    respy_obj.write_out()

    # Iterate over the grid of requested slaves.
    for num_slaves in grid_slaves:
        dirname = '{:}'.format(num_slaves)
        os.mkdir(dirname), os.chdir(dirname)

        respy_obj.unlock()
        respy_obj.set_attr('num_procs', num_slaves + 1)
        if num_slaves > 1:
            respy_obj.set_attr('is_parallel', True)
        else:
            respy_obj.set_attr('is_parallel', False)
        respy_obj.lock()
        respy_obj.write_out()

        start_time = datetime.now()
        respy.estimate(respy_obj)
        finish_time = datetime.now()

        os.chdir('../')

        record_information(start_time, finish_time, num_slaves)

    os.chdir('../')


def record_information(start_time, finish_time, num_slaves):
    fmt = '{:>15} {:>25} {:>25} {:>15}\n'
    if not os.path.exists('scalability.respy.info'):
        with open('scalability.respy.info', 'a') as out_file:
            out_file.write('\n Time\n\n')
            out_file.write(
                fmt.format(*['Slaves', 'Start', 'Stop', 'Duration']))
            out_file.write('\n')

    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    finish_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")

    duration_time = finish_time - start_time
    duration_str = strfdelta(duration_time, "{H:02}:{M:02}:{S:02}")

    with open('scalability.respy.info', 'a') as out_file:
        line = [num_slaves, start_str, finish_str, duration_str]
        out_file.write(fmt.format(*line))


