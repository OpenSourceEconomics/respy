"""This module includes code to configure pytest."""


def pytest_addoption(parser):
    """Add a custom option to the pytest call.

    Simply use

    .. code-block:: bash

        $ pytest --n-random-tests=n

    to restrict running each random test with `n` increments of the base seed given by
    `pytest-randomly`.

    """
    parser.addoption(
        "--n-random-tests",
        action="store",
        default=5,
        help="Number of runs for each random test with different seeds.",
    )


def pytest_generate_tests(metafunc):
    """Re-run some tests with different seeds by incrementing the base seed.

    The base seed is given by `pytest-randomly` in each session derived from the
    timestamp. You can use five as the seed value with

    .. code-block:: bash

        $ pytest --randomly-seed=5

    Then, tests with certain parameter names are parameterized with incremented seed
    values (5, 6, 7, 8, ...).

    """
    if "model_or_seed" in metafunc.fixturenames:
        n_random_tests = int(metafunc.config.getoption("--n-random-tests"))
        seeds = [
            metafunc.config.getoption("--randomly-seed") + i
            for i in range(n_random_tests)
        ]

        mark = metafunc.definition.get_closest_marker("parametrize")
        if mark:
            mark.args[1].extend(seeds)
        else:
            metafunc.parametrize("model_or_seed", seeds)
