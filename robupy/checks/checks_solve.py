""" This module contains some additional checks related to the solution of the dynamic programming model.
"""

# standard library
import numpy as np
import pandas as pd


def checks_solve(str_, robupy_obj, *args):
    """ This checks the integrity of the objects related to the solution of the model.
    """

    # Distribute class attributes
    if robupy_obj is not None:

        num_periods = robupy_obj.get_attr('num_periods')

        edu_start = robupy_obj.get_attr('edu_start')

        edu_max = robupy_obj.get_attr('edu_max')

        states_all = robupy_obj.get_attr('states_all')

        states_number_period = robupy_obj.get_attr('states_number_period')


    if str_ == '_wrapper_create_state_space_out':

        # Distribute input parameters
        states_all, states_number_period, mapping_state_idx, = args

        # If the agent never increased their level of education,
        # the lagged education variable cannot take a value
        # larger than zero.
        for period in range(1, num_periods):
            indices = (np.where(states_all[period, :, :][:, 2] == 0))
            for index in indices:
                assert (np.all(states_all[period, :, :][index, 3]) == 0)

        # No values can be larger than constraint time.
        # The exception in the lagged schooling variable
        # in the first period, which takes value one but has
        # index zero.
        for period in range(num_periods):
            assert (np.nanmax(states_all[period, :, :3]) <= period)

        # Lagged schooling can only take value zero or one if finite.
        # In fact, it can only take value one in the first period.
        for period in range(num_periods):
            assert (np.all(states_all[0, :, 3]) == 1)
            assert (np.nanmax(states_all[period, :, 3]) == 1)
            assert (np.nanmin(states_all[period, :, :3]) == 0)

        # All finite values have to be larger or equal to zero.
        # The loop is required as np.all evaluates to FALSE
        # for this condition (see NUMPY documentation).
        for period in range(num_periods):
            assert (
                np.all(states_all[period, :states_number_period[period]] >= 0))

        # The maximum number of additional education years is never
        # larger than (EDU_MAX - EDU_START).
        for period in range(num_periods):
            assert (np.nanmax(states_all[period, :, :][:, 2], axis=0) <= (
                edu_max - edu_start))

        # Check for duplicate rows in each period
        for period in range(num_periods):
            assert (np.sum(pd.DataFrame(
                states_all[period, :states_number_period[period],
                :]).duplicated()) == 0)

        # Checking validity of state space values. All valid
        # values need to be finite.
        for period in range(num_periods):
            assert (np.all(
                np.isfinite(states_all[period, :states_number_period[period]])))

        # There are no infinite values in final period.
        assert (np.all(np.isfinite(states_all[(num_periods - 1), :, :])))

        # There are is only one finite realization in period one.
        assert (np.sum(np.isfinite(mapping_state_idx[0, :, :, :, :])) == 1)

        # If valid, the number of state space realizations in period two is four.
        if num_periods > 1:
            assert (np.sum(np.isfinite(mapping_state_idx[1, :, :, :, :])) == 4)

        # Check that mapping is defined for all possible realizations
        # of the state space by period. Check that mapping is not defined
        # for all inadmissible values.
        is_infinite = np.tile(False, reps=mapping_state_idx.shape)
        for period in range(num_periods):
            # Subsetting valid indices
            indices = states_all[period, :states_number_period[period], :]
            for index in indices:
                # Check for finite value at admissible state
                assert (np.isfinite(mapping_state_idx[
                                        period, index[0], index[1], index[2],
                                        index[3]]))
                # Record finite value
                is_infinite[
                    period, index[0], index[1], index[2], index[3]] = True
        # Check that all admissible states are finite
        assert (np.all(np.isfinite(mapping_state_idx[is_infinite == True])))
        # Check that all inadmissible states are infinite
        assert (
            np.all(np.isfinite(mapping_state_idx[is_infinite == False])) == False)

    # Check the ex ante period payoffs
    elif str_ == 'periods_payoffs_ex_ante':

        # Distribute input parameters
        states_all, states_number_period, periods_payoffs_ex_ante = args

        # Check that the payoffs are finite for all admissible values and infinite for all others.
        is_infinite = np.tile(False, reps=periods_payoffs_ex_ante.shape)
        for period in range(num_periods):
            # Loop over all possible states
            for k in range(states_number_period[period]):
                # Check that wages are all positive
                assert (np.all(periods_payoffs_ex_ante[period, k, :2] > 0.0))
                # Check for finite value at admissible state
                assert (
                    np.all(np.isfinite(periods_payoffs_ex_ante[period, k, :])))
                # Record finite value
                is_infinite[period, k, :] = True
            # Check that all admissible states are finite
            assert (
                np.all(np.isfinite(periods_payoffs_ex_ante[is_infinite ==
                                                           True])))
            # Check that all inadmissible states are infinite
            assert (np.all(np.isfinite(
                periods_payoffs_ex_ante[is_infinite == False])) == False)

    # Check the ex ante period payoffs and its ingredients
    elif str_ == 'periods_emax':

        # Distribute input parameters
        emax, future_payoffs = args

        # Check that the payoffs are finite for all admissible values and infinite for all others.
        is_infinite = np.tile(False, reps=emax.shape)
        for period in range(num_periods):
            # Loop over all possible states
            for k in range(states_number_period[period]):
                # Check for finite value at admissible state
                assert (np.all(np.isfinite(emax[period, k])))
                # Record finite value
                is_infinite[period, k] = True
            # Check that all admissible states are finite
            assert (np.all(np.isfinite(emax[is_infinite == True])))
            # Check that all inadmissible states are infinite
            if num_periods == 1:
                assert (len(emax[is_infinite == False]) == 0)
            else:
                assert (
                    np.all(np.isfinite(emax[is_infinite == False])) == False)

        # Check that the payoffs are finite for all admissible values and infinite for all others.
        for period in range(num_periods - 1):
            # Loop over all possible states
            for k in range(states_number_period[period]):
                # Check for finite value at admissible state, infinite
                # values are allowed for the third column when the
                # maximum level of education is attained.
                assert (np.all(np.isfinite(future_payoffs[period, k, :2])))
                assert (np.all(np.isfinite(future_payoffs[period, k, 3])))
                # Special checks for infinite value due to
                # high education.
                if not np.isfinite(future_payoffs[period, k, 2]):
                    assert (states_all[period, k][2] == edu_max - edu_start)

    else:

        raise AssertionError

    # Finishing
    return True
