

def record_warning_crit_val(count):
    """ Sometimes the value of the criterion function is too extreme for
    pretty printing in the output files. This warning indicates that this is
    in fact the case. Internally, the original values are all used.
    """
    with open('est.respy.log', 'a') as out_file:

        line = '   Warning: '

        if count == 0:
            line += 'Starting '
        if count == 1:
            line += 'Step '
        if count == 2:
            line += 'Current '

        line += 'value of criterion function too large to write to file, ' \
                'internals unaffected.\n'

        out_file.write(line)
