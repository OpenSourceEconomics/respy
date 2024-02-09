from itertools import count

import pytest

from respy.interpolate import _split_interpolation_points_evenly
from respy.solve import get_solve_func
from respy.tests.utils import process_model_or_seed

@pytest.mark.skip
@pytest.mark.end_to_end
def test_run_through_of_solve_with_interpolation(seed):
    params, options = process_model_or_seed(
        seed, point_constr={"n_periods": 5, "interpolation_points": 10}
    )

    solve = get_solve_func(params, options)
    solve(params)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dense_index_to_n_states, interpolation_points",
    [({0: 50, 1: 150}, 100), ({0: 4, 5: 4, 10: 4}, 10)],
)
def test_split_interpolation_points_evenly(
    dense_index_to_n_states, interpolation_points
):
    options = {
        "interpolation_points": interpolation_points,
        "solution_seed_iteration": count(0),
    }

    interpolations_points_splitted = _split_interpolation_points_evenly(
        dense_index_to_n_states, 0, options
    )

    for index in dense_index_to_n_states:
        assert dense_index_to_n_states[index] >= interpolations_points_splitted[index]
