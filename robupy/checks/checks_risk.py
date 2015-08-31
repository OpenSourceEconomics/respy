""" This module contains some additional checks related to solving the
model under the presence of risk
"""

# standard library
import numpy as np
import pandas as pd


def checks_risk(str_, *args):
    """ This checks the integrity of the objects related to the solution of
    the model under risk.
    """

    if str_ == 'get_payoffs_risk':

        # Distribute arguments
        ambiguity_args, = args

        # Checks
        assert (ambiguity_args is None)

    else:

        raise AssertionError

    # Finishing
    return True