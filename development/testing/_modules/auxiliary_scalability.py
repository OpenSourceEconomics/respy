from datetime import datetime
import datetime as dt
import shlex
import os

from auxiliary_shared import strfdelta
from config import SPEC_DIR

import respy


def get_actual_evaluations():

    with open('est.respy.info', 'r') as infile:
        for line in infile.readlines():
            list_ = shlex.split(line)

            if not list_:
                continue

            if not len(list_) == 4:
                continue

            if list_[2] == 'Evaluations':
                return int(list_[3])

    raise AssertionError


def run(spec_dict, fname, grid_slaves):
    dirname = fname.replace('.ini', '')

    os.mkdir(dirname)
    os.chdir(dirname)

    respy_obj = respy.RespyCls(SPEC_DIR + fname)

    respy_obj.unlock()
    respy_obj.set_attr('is_debug', False)

    respy_obj.set_attr('file_est', '../data.respy.dat')
    for key_ in spec_dict.keys():
        respy_obj.set_attr(key_, spec_dict[key_])
    respy_obj.lock()

    maxfun = respy_obj.get_attr('maxfun')
    min_slave = min(grid_slaves)

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

        if num_slaves == min_slave:
            duration_baseline = finish_time - start_time
            num_evals = get_actual_evaluations()

        os.chdir('../')

        record_information(start_time, finish_time, num_slaves, maxfun,
                           duration_baseline, num_evals, min_slave)

    os.chdir('../')


def record_information(start_time, finish_time, num_slaves, maxfun,
                       duration_baseline, num_evals, min_slave):
    fmt = '{:>15} {:>25} {:>25} {:>15} {:>15} {:>15}\n'
    if not os.path.exists('scalability.respy.info'):
        with open('scalability.respy.info', 'a') as out_file:
            out_file.write('\n    Benchmarking a maximum of ' + str(
                maxfun) + ' evaluations\n\n')
            out_file.write(
                fmt.format(*['Slaves', 'Start', 'Stop', 'Duration',
                             'Benchmark', 'Evaluations']))
            out_file.write('\n')

    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    finish_str = finish_time.strftime("%Y-%m-%d %H:%M:%S")

    duration_time = finish_time - start_time
    duration_actual_str = strfdelta(duration_time, "{H:02}:{M:02}:{S:02}")

    duration_linear_str = '---'
    if not num_slaves == min_slave:
        duration_linear_secs = duration_baseline.total_seconds() / (
            num_slaves / max(min_slave, 1))
        duration_linear = dt.timedelta(seconds=duration_linear_secs)
        duration_linear_str = strfdelta(duration_linear, "{H:02}:{M:02}:{S:02}")

    with open('scalability.respy.info', 'a') as out_file:
        line = [num_slaves, start_str, finish_str, duration_actual_str,
                duration_linear_str, num_evals]
        out_file.write(fmt.format(*line))


