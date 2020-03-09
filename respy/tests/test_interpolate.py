from respy.solve import get_solve_func
from respy.tests.random_model import generate_random_model


def test_run_through_of_solve_with_interpolation(seed):  # noqa: U100, U101
    params, options = generate_random_model(
        point_constr={"n_periods": 5, "interpolation_points": 10}
    )

    solve = get_solve_func(params, options)
    solve(params)
