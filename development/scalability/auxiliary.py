from datetime import datetime
from string import Formatter
import socket
import glob
import sys
import os

import respy

sys.path.insert(0, '../modules')
from clsMail import MailCls


def strfdelta(tdelta, fmt):
    f, d = Formatter(), {}
    l = {'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    k = map(lambda x: x[1], list(f.parse(fmt)))
    rem = int(tdelta.total_seconds())
    for i in ('D', 'H', 'M', 'S'):
        if i in k and i in l.keys():
            d[i], rem = divmod(rem, l[i])
    return f.format(fmt, **d)


def process_tasks(spec_dict, fname, grid_slaves):
    dirname = fname.replace('.ini', '')
    os.mkdir(dirname), os.chdir(dirname)

    # Distribute details about specification
    optimizer_options = spec_dict['optimizer_options']
    optimizer_used = spec_dict['optimizer_used']
    num_draws_emax = spec_dict['num_draws_emax']
    num_draws_prob = spec_dict['num_draws_prob']
    num_agents = spec_dict['num_agents']
    scaling = spec_dict['scaling']
    maxfun = spec_dict['maxfun']

    respy_obj = respy.RespyCls('../' + fname)

    respy_obj.unlock()
    respy_obj.set_attr('file_est', '../data.respy.dat')
    respy_obj.set_attr('optimizer_options', optimizer_options)
    respy_obj.set_attr('optimizer_used', optimizer_used)
    respy_obj.set_attr('num_draws_emax', num_draws_emax)
    respy_obj.set_attr('num_draws_prob', num_draws_prob)
    respy_obj.set_attr('num_agents_est', num_agents)
    respy_obj.set_attr('num_agents_sim', num_agents)
    respy_obj.set_attr('scaling', scaling)
    respy_obj.set_attr('maxfun', maxfun)
    respy_obj.lock()

    if 'num_periods' in spec_dict.keys():
        respy_obj.unlock()
        respy_obj.set_attr('num_periods', spec_dict['num_periods'])
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


def send_notification():
    """ Finishing up a run of the testing battery.
    """
    hostname = socket.gethostname()
    subject = ' RESPY: Monte Carlo Exercise '
    message = ' The scalability exercise is completed on @' + hostname + '.'

    mail_obj = MailCls()
    mail_obj.set_attr('subject', subject)
    mail_obj.set_attr('message', message)
    mail_obj.lock()

    mail_obj.send()


def aggregate_information():
    dirnames = []
    for fname in glob.glob('*.ini'):
        dirnames += [fname.replace('.ini', '')]
    with open('scalability.respy.info', 'w') as outfile:
        outfile.write('\n')
        for dirname in dirnames:
            outfile.write(' ' + dirname + '\n')
            os.chdir(dirname)
            with open('scalability.respy.info', 'r') as infile:
                outfile.write(infile.read())
            os.chdir('../')
            outfile.write('\n\n')
