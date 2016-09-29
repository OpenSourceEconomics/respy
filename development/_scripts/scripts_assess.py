#!/usr/bin/env python
from __future__ import print_function

from statsmodels.tools.eval_measures import rmse
import argparse
import os

from auxiliary_reliability import get_choice_probabilities


def dist_input_arguments(parser):
    """ Check input for estimation script.
    """
    # Parse arguments
    args = parser.parse_args()

    # Distribute arguments
    base= args.base
    alt = args.alt

    # Check attributes
    for fname in [alt, base]:
        assert (os.path.exists(fname))

    # Finishing
    return base, alt


def run(base, alt):

    probs_base = get_choice_probabilities(base, is_flatten=True)
    probs_alt = get_choice_probabilities(alt, is_flatten=True)
    stat = rmse(probs_base, probs_alt)
    str_ = '\n The RMSE amounts to {:5.4f}.\n'
    print(str_.format(stat))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
        'Calculate the root-mean-square error (RMSE).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--base', action='store', dest='base',
        help='baseline information', required=True)

    parser.add_argument('--alt', action='store', dest='alt',
        help='alternative information', required=True)

    # Run estimation
    run(*dist_input_arguments(parser))
