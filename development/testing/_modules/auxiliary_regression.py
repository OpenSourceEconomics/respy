import shlex


def write_request(num_tests):
    # Transfer details of request. We cannot use PICKLE as the information
    # need to be accessible from PYTHON2 and PYTHON3 at the same time.
    with open('request.txt', 'w') as out_file:
        out_file.write(str(num_tests))


def read_request():

    rslt = list()
    with open('request.txt', 'r') as out_file:
        for line in out_file.readlines():
            rslt = int(shlex.split(line)[0])

    return rslt


def dist_input_arguments(parser):
    """ Check input for script.
    """

    request = parser.parse_args().request
    num_tests = parser.parse_args().num_tests
    version = parser.parse_args().version

    assert isinstance(num_tests, int)
    assert (0 < num_tests)

    if version is not None:
        assert version in [2, 3]

    return request, num_tests, version
