

def log_warning_crit_val(count):

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
