

def start_logging():
    """ Start logging of performance.
    """

    # Initialize logger
    logger = logging.getLogger('DEV-TEST')
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler('logging.log', 'w')
    fh.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(' %(asctime)s     %(message)s \n', datefmt='%I:%M:%S %p')
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)

def distribute_input(parser):
    """ Check input for estimation script.
    """
    # Parse arguments.
    args = parser.parse_args()

    # Distribute arguments.
    hours = args.hours
    notification = args.notification

    # Assertions.
    assert (notification in [True, False])
    assert (isinstance(hours, float))
    assert (hours > 0.0)

    # Validity checks
    if notification:
        # Check that the credentials file is stored in the user's HOME directory.
        assert (os.path.exists(os.environ['HOME'] + '/.credentials'))

    # Finishing.
    return hours, notification

def finish(dict_, HOURS, notification):
    """ Finishing up a run of the testing battery.
    """
    # Antibugging.
    assert (isinstance(dict_, dict))
    assert (notification in [True, False])

    # Auxiliary objects.
    hostname = socket.gethostname()

    with open('logging.log', 'a') as file_:

        file_.write(' Summary \n\n')

        str_ = '   Test {0:<10} Success {1:<10} Failures  {2:<10}\n'

        for label in sorted(dict_.keys()):

            success = dict_[label]['success']

            failure = dict_[label]['failure']

            file_.write(str_.format(label, success, failure))

        file_.write('\n')

    if notification:

        subject = ' ROBUPY: Completed Testing Battery '

        message = ' A ' + str(HOURS) +' hour run of the testing battery on @' + \
                  hostname + ' is completed.'

        mail_obj = MailCls()

        mail_obj.set_attr('subject', subject)

        mail_obj.set_attr('message', message)

        mail_obj.set_attr('attachment', 'logging.log')

        mail_obj.lock()

        mail_obj.send()


def cleanup():
    """ Cleanup after test battery.
    '"""

    files = []

    ''' Clean main.
    '''
    files += glob.glob('.waf*')

    files += glob.glob('.pkl*')

    files += glob.glob('.txt*')

    files += glob.glob('.grm.*')

    files += glob.glob('*.pkl')

    files += glob.glob('*.txt')

    files += glob.glob('*.out')

    files += glob.glob('test*.ini')

    files += glob.glob('*.pyc')

    files += glob.glob('.seed')

    files += glob.glob('.dat*')

    files += glob.glob('*.robupy.*')

    files += glob.glob('*.robufort.*')

    ''' Clean modules.
        '''
    files += glob.glob('modules/*.out*')

    files += glob.glob('modules/*.pyc')

    files += glob.glob('modules/*.mod')

    files += glob.glob('modules/dp3asim')

    files += glob.glob('modules/lib')

    files += glob.glob('modules/include')

    files += glob.glob('modules/*.so')

    files += glob.glob('*.dat')

    files += glob.glob('.restud.testing.scratch')

    files += glob.glob('modules/*.o')

    files += glob.glob('modules/__pycache__')

    for file_ in files:

        try:

            os.remove(file_)

        except:

            try:

                shutil.rmtree(file_)

            except:

                pass

def check_ambiguity_optimization():
    """ This function checks that less than 5% of all optimization for each
    period fail.
    """
    def _process_cases(list_):
        """ Process cases and determine whether keyword or empty line.
        """
        # Antibugging
        assert (isinstance(list_, list))

        # Get information
        is_empty = (len(list_) == 0)

        if not is_empty:
            is_summary = (list_[0] == 'SUMMARY')
        else:
            is_summary = False

        # Antibugging
        assert (is_summary in [True, False])
        assert (is_empty in [True, False])

        # Finishing
        return is_empty, is_summary

    is_relevant = False

    # Check relevance
    if not os.path.exists('ambiguity.robupy.log'):
        return

    for line in open('ambiguity.robupy.log').readlines():

        # Split line
        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_summary = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        # Prepare dictionary
        if is_summary:
            is_relevant = True
            continue

        if not is_relevant:
            continue

        if list_[0] == 'Period':
            continue

        period, total, success, failure = list_

        total = success + failure

        if float(failure)/float(total) > 0.05:
            raise AssertionError