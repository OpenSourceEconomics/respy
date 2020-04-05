from respy.solve import get_solve_func
from respy.tests.utils import process_model_or_seed


def test_run_through_of_solve_with_interpolation(seed):
    params, options = process_model_or_seed(
        seed, point_constr={"n_periods": 5, "interpolation_points": 10}
    )

    solve = get_solve_func(params, options)
    solve(params)
