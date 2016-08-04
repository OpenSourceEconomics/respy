import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from datetime import timedelta

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

def ylabel_formatting(x, _):
    d = timedelta(seconds=x)
    return str(d)

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

