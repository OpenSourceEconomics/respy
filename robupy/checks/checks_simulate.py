""" This module contains some additional checks related to the simulation of
the model..
"""

# standard library
import numpy as np
import pandas as pd


def checks_simulate(str_, robupy_obj, *args):
    """ This checks the integrity of the objects related to the solution of
    the model.
    """

    # Distribute class attributes
    num_periods = robupy_obj.get_attr('num_periods')
    num_agents = robupy_obj.get_attr('num_agents')

    edu_start = robupy_obj.get_attr('edu_start')

    edu_max = robupy_obj.get_attr('edu_max')

    if str_ == 'data_frame':

        # Distribute input parameters
        data_frame, = args

        # Check dimension of data frame
        assert (data_frame.shape == (num_periods * num_agents, 8))

        # Check that there are the appropriate number of values
        for i in range(8):
            # This is not applicable for the wage observations
            if i == 3:
                continue
            # Check valid values
            assert (data_frame.count(axis=0)[i] == (num_periods * num_agents))

        # Check that the maximum number of values are valid
        for i, max_ in enumerate(
                [num_agents, num_periods, 3, num_periods, num_periods,
                 num_periods,
                 edu_max, 1]):
            # Agent and time index
            if i in [0, 1]:
                assert (data_frame.max(axis=0)[i] == (max_ - 1))
            # Choice observation
            if i == 2:
                assert (data_frame.max(axis=0)[i] <= max_)
            # Wage observations
            if i == 3:
                pass
            # Work experience A, B
            if i in [4, 5]:
                assert (data_frame.max(axis=0)[i] <= max_)
            # Education
            if i in [6]:
                assert (data_frame.max(axis=0)[i] <= max_)
            # Lagged Education
            if i in [7]:
                assert (data_frame.max(axis=0)[i] <= max_)

        # Each valid period indicator occurs as often as
        # agents in the dataset.
        for period in range(num_periods):
            assert (data_frame[1].value_counts()[period] == num_agents)

        # Each valid agent indicator occurs as often
        # as periods in the dataset.
        for agent in range(num_agents):
            assert (data_frame[0].value_counts()[agent] == num_periods)

        # Check valid values of wage observations
        for count in range(num_agents * num_periods):
            is_working = (data_frame[2][count] in [0, 1])
            if is_working:
                assert (np.isfinite(data_frame[3][count]))
                assert (data_frame[3][count] > 0.00)
            else:
                assert (np.isfinite(data_frame[3][count]) == False)

    else:

        raise AssertionError

    # Finishing
    return True
