""" Collect some auxiliary functions.
"""

# standard library
import numpy as np


def create_simulation_grid(num_points, states_all):
    """ Create a grid for the points of random integration.
    """
    # Checks
    assert (isinstance(numn_points, int))
    # Initialize grid for simulation points
    simulation_grid = np.tile(np.nan, num_points)

    # Extract total number of states.
    num_states = states_number_period[period]
    print(num_states)
    import sys
    sys.exit("inside")
    # Draw a random subset of points when the number of possible states
    # is larger then the requested number of interpolation points.
    if num_points > num_states:
        subset = np.random.choice(states_number_period[period],
                                          size=num_points, replace=False)
    # Fill up simulation grid
    simulation_grid[i, :min(num_states, num_points)] = subset

    # Checks
    assert (num_points <= max_states_period)
    assert (simulation_grid.shape == (num_periods, num_points))
    assert (np.all(np.isfinite(simulation_grid[num_periods - 1, :])))
    for i, period in enumerate(range(num_periods)):
            assert(len(simulation_grid[i, :]) == len(set(simulation_grid[i, :])))

    # Finishing
    return simulation_grid
