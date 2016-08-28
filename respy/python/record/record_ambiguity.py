import shlex


def record_ambiguity(period, k, x_shift, div, success, message):
    """ Write result of optimization problem to log file.
    """

    with open('amb.respy.log', 'a') as file_:

        string = ' PERIOD{0[0]:>7}  STATE{0[1]:>7}\n\n'
        file_.write(string.format([period, k]))

        string = '   {:<1}{:+10.5f}\n'
        args = ['A'] + [x_shift[0]]
        file_.write(string.format(*args))

        string = '   {:<1}{:+10.5f}\n\n'
        args = ['B'] + [x_shift[1]]
        file_.write(string.format(*args))

        string = '   {:<15}{:<15.5f}\n'
        file_.write(string.format(*['Divergence', div]))

        string = '   {:<15}{:<25}\n'
        file_.write(string.format(*['Success', str(success)]))

        string = '   {:<15}{:<25}\n\n\n'
        file_.write(string.format(*['Message', message]))

    # Summarize the overall performance.
    if period == 0:
        record_ambiguity_summary()

def record_ambiguity_summary():
    """ Summarize optimizations in case of ambiguity.
    """

    def _process_cases(list_internal):
        """ Process cases and determine whether keyword or empty line.
        """
        # Antibugging
        assert (isinstance(list_internal, list))

        # Get information
        is_empty_internal = (len(list_internal) == 0)

        if not is_empty_internal:
            is_block_internal = list_internal[0] == 'PERIOD'
        else:
            is_block_internal = False

        # Antibugging
        assert (is_block_internal in [True, False])
        assert (is_empty_internal in [True, False])

        # Finishing
        return is_empty_internal, is_block_internal

    # Distribute class attributes
    dict_ = dict()

    for line in open('amb.respy.log').readlines():

        # Split line
        list_ = shlex.split(line)

        # Determine special cases
        is_empty, is_block = _process_cases(list_)

        # Applicability
        if is_empty:
            continue

        # Prepare dictionary
        if is_block:

            period = int(list_[1])

            if period in dict_.keys():
                continue

            dict_[period] = {}
            dict_[period]['success'] = 0
            dict_[period]['failure'] = 0
            dict_[period]['total'] = 0

        # Collect success indicator
        if list_[0] == 'Success':
            dict_[period]['total'] += 1

            is_success = (list_[1] == 'True')
            if is_success:
                dict_[period]['success'] += 1
            else:
                dict_[period]['failure'] += 1

    with open('amb.respy.log', 'a') as file_:

        file_.write(' SUMMARY\n\n')

        string = '''{0[0]:>10} {0[1]:>10} {0[2]:>10} {0[3]:>10}\n'''
        args = string.format(['Period', 'Total', 'Success', 'Failure'])
        file_.write(args)

        file_.write('\n')

        for period in range(max(dict_.keys()) + 1):
            total = dict_[period]['total']

            success = dict_[period]['success'] / total
            failure = dict_[period]['failure'] / total

            string = '''{0[0]:>10} {0[1]:>10} {0[2]:10.2f} {0[3]:10.2f}\n'''
            file_.write(string.format([period, total, success, failure]))

        file_.write('\n')


