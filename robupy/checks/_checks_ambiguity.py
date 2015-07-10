""" This module contains some additional checks related to the
    solution of the dynamic programming problem with ambiguity.
"""

# standard library
import numpy as np

def _checks(str_, *args):
    """ This checks the integrity of the objects related to the
        solution of the model.
    """

    if str_ == '_get_start':

        # Distribute input parameters
        x0,  = args

        # Check quality of starting values
        assert (len(x0) == 4)
        assert (np.all(np.isfinite(x0)))

        assert (all(val == 0 for val in x0))

    elif str_ == 'simulate_emax_ambiguity':

        # Distribute input parameters
        simulated, opt = args

        # Check quality of results. As I evaluate the function at the parameters
        # resulting from the optimization, the value of the criterion function
        # should be the same.
        assert (simulated == opt['fun'])

    elif str_ == '_criterion':

        # Distribute input parameters
        simulated, = args

        # Check quality of bounds
        assert (np.isfinite(simulated))

    elif str_ == '_prep_absolute':

        # Distribute input parameters
        bounds, = args

        # Check quality of bounds
        assert (len(bounds) == 4)

        for i in range(4):
            assert (bounds[0] == bounds[i])

    else:

        raise AssertionError

    # Finishing
    return True