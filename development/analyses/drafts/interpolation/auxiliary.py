""" Collect some auxiliary functions.
"""

# standard library
import numpy as np


def create_simulation_grid(num_points, max_states_period,
                           states_number_period, num_periods):
    """ Create the simulation grid.
    """

    simulation_grid = None

    if num_points < max_states_period:
        # Initialize simulation grid
        simulation_grid = np.tile(np.nan, (num_periods, num_points))
        # Loop over all states and draw random interpolation points
        for i, period in enumerate(range(num_periods)):
            # Extract total number of states.
            num_states = states_number_period[period]
            # Draw a random subset of points when the number of possible states is
            # larger then the requested number of interpolation points.
            if num_points > num_states:
                subset = range(states_number_period[period])
            else:
                subset = np.random.choice(states_number_period[period], size=num_points,
                                          replace=False)
            # Fill up simulation grid
            simulation_grid[i, :min(num_states, num_points)] = subset

        # Check
        assert (num_points <= max_states_period)
        assert (simulation_grid.shape == (num_periods, num_points))
        assert (np.all(np.isfinite(simulation_grid[num_periods - 1, :])))
        for i, period in enumerate(range(num_periods)):
            assert(len(simulation_grid[i, :]) == len(set(simulation_grid[i, :])))

    # Finishing
    return simulation_grid