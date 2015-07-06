""" This module contains some additional checks related to soliving the
model under the presence of risk
"""

# standard library
import numpy as np
import pandas as pd


def _checks(str_, *args):
    """ This checks the integrity of the objects related to the solution of
    the model under risk.
    """

    if str_ == 'simulate_emax_risk':

        # Distribute arguments
        ambigutiy_args, = args

        # Checks
        assert (ambigutiy_args is None)

    else:

        raise AssertionError

    # Finishing
    return True