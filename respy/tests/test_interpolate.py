from respy.solve import get_solve_func
from respy.tests.random_model import generate_random_model


def test_simple_run():
    params, options = generate_random_model(
        point_constr={"n_periods": 5, "interpolation_points": 10}
    )

    solve = get_solve_func(params, options)
    solve(params)
