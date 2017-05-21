from datetime import datetime
import datetime as dt
import shlex
import os

from auxiliary_shared import update_class_instance
from auxiliary_shared import aggregate_information
from auxiliary_shared import send_notification
from auxiliary_shared import strfdelta
from auxiliary_shared import cleanup
from config import SPEC_DIR

import respy


def run(spec_dict):
    """ Details of the scalability exercise can be specified in the code block below. Note that 
    only deviations from the benchmark initialization files need to be addressed.
    """

    cleanup()

    os.mkdir('rslt')
    os.chdir('rslt')

    grid_slaves = spec_dict['slaves']
    for fname in spec_dict['fnames']:
        run_single(spec_dict, fname, grid_slaves)

    aggregate_information('scalability', spec_dict['fnames'])

    send_notification('scalability')
    os.chdir('../')


def run_single(spec_dict, fname, grid_slaves):
    """ Run an estimation task that allows to get a sense of the scalability of the code.
    """
    os.mkdir(fname.replace('.ini', ''))
    os.chdir(fname.replace('.ini', ''))

    # Read in the baseline specification.
    respy_obj = respy.RespyCls(SPEC_DIR + fname)

    # We remove all bounds as this makes it easier to request static models for example with an
    # initialization file for a risk model. Usually, there are bounds on the discount rate that
    # will raise an error otherwise.
    optim_paras = respy_obj.get_attr('optim_paras')
    optim_paras['paras_bounds'][0] = [0.00, None]
    optim_paras['paras_bounds'][1] = [0.00, None]

    # Now we update the class instance with the details of the request.
    update_class_instance(respy_obj, spec_dict)
    min_slave = min(grid_slaves)

    # Simulate the baseline dataset, which is used regardless of the number of slaves. We will
    # use the largest processor count for this step.
    respy_obj.unlock()
    respy_obj.set_attr('num_procs', max(grid_slaves))
    respy_obj.lock()

    respy_obj.write_out()
    respy.simulate(respy_obj)

    # Iterate over the grid of requested slaves.
    for num_slaves in grid_slaves:
        dirname = '{:}'.format(num_slaves)

        os.mkdir(dirname)
        os.chdir(dirname)

        respy_obj.unlock()
        respy_obj.set_attr('num_procs', num_slaves + 1)
        respy_obj.lock()

        respy_obj.write_out()

        respy.estimate(respy_obj)

        # Get results from the special output file that is integrated in the RESPY package just
        # for the purpose of scalability exercises.
        with open('.scalability.respy.log') as in_file:
            rslt = []
            for line in in_file.readlines():

                list_ = shlex.split(line)
                str_ = list_[1] + ' ' + list_[2]
                rslt += [datetime.strptime(str_, "%d/%m/%Y %H:%M:%S")]

        start_time, finish_time = rslt

        if num_slaves == min_slave:
            duration_baseline = finish_time - start_time

        os.chdir('../')

        args = [start_time, finish_time, num_slaves]
        args += [duration_baseline, min_slave]

        record_information(*args)

    os.chdir('../')


def record_information(start_time, finish_time, num_slaves, duration_baseline, min_slave):
    """ Record the information on execution time, which involves a lot of formatting of different data types.
    """
    fmt = '{:>15} {:>25} {:>25} {:>15} {:>15}\n'
    if not os.path.exists('scalability.respy.info'):
        with open('scalability.respy.info', 'a') as out_file:
            out_file.write('\n')
            out_file.write(
                fmt.format(*['Slaves', 'Start', 'Stop', 'Duration',
                             'Benchmark']))
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
                duration_linear_str]
        out_file.write(fmt.format(*line))
