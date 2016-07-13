def record_warning(count):
    """ Sometimes the value of the criterion function is too extreme for
    pretty printing in the output files. This warning indicates that this is
    in fact the case. Internally, the original values are all used. The
    numbering of warning messages is aligned with FORTRAN and thus starts at
    one and not zero.
    """
    with open('est.respy.log', 'a') as out_file:

        if count == 1:
            msg = 'Starting value of criterion function too large to write ' \
                  'to file, internals unaffected.'
        elif count == 2:
            msg = 'Step value of criterion function too large to write ' \
                  'to file, internals unaffected.'
        elif count == 3:
            msg = 'Current value of criterion function too large to write ' \
                  'to file, internals unaffected.'
        elif count == 4:
            msg = 'Stabilization of otherwise zero element on diagonal of Cholesky decomposition.'
        elif count == 5:
            msg = 'Some agents have a numerically zero probability, stabilization of logarithm required.'
        else:
            raise AssertionError

        out_file.write('   Warning: ' + msg + '\n')

