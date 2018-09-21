#!/usr/bin/env python
import argparse
import os

from respy.custom_exceptions import UserError
from respy import RespyCls


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    init_file = args.init_file
    single = args.single

    # Check attributes
    assert (single in [True, False])
    if not os.path.exists(init_file):
        raise UserError('Initialization file does not exist')

    # Finishing
    return single, init_file


def scripts_estimate(single, init_file):
    """ Wrapper for the estimation.
    """
    # Read in baseline model specification.
    respy_obj = RespyCls(init_file)

    # Set maximum iteration count when only an evaluation of the criterion function is requested.
    if single:
        respy_obj.unlock()
        respy_obj.set_attr('maxfun', 0)

        precond_spec = dict()
        precond_spec['type'] = 'identity'
        precond_spec['minimum'] = 0.01
        precond_spec['eps'] = 0.01

        respy_obj.set_attr('precond_spec', precond_spec)

        respy_obj.lock()

    # Optimize the criterion function.
    respy_obj.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start of estimation run with the RESPY package.')

    parser.add_argument('--single', action='store_true', dest='single', default=False,
                        help='single evaluation')

    parser.add_argument('--init', action='store', dest='init_file',
                        default='model.respy.ini', help='initialization file')

    # Process command line arguments
    args = dist_input_arguments(parser)

    # Run estimation
    scripts_estimate(*args)
