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

