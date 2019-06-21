Randomness
==========

``respy`` embraces randomness to study individual behavior under risk. At the same time,
it is crucial to make results reproducible. In the following, all parts of the
implementation which rely on randomness are documented.

shared
------

.. currentmodule:: respy.shared

The function :func:`create_base_draws` is used in all parts, solution,
simulation, and estimation, to generate random draws.


.. autosummary::
    :toctree: ../_generated/

    create_base_draws

solve
-----

.. currentmodule:: respy.solve

Routines under ``respy.solve`` use the seed specified in the
``options["solution_seed"]`` to control randomness. Apart from the draws,
:func:`~respy.solve.solve` relies on the following function.


.. autosummary::
    :toctree: ../_generated/

    get_not_interpolated_indicator

simulate
--------

Routines under ``respy.simulate`` use the seed specified in the
``options["simulation_seed"]`` to control randomness. Apart from the draws,
:func:`~respy.simulate.simulate` relies on the following three functions to generate
starting values for simulated agents.

.. currentmodule:: respy.simulate

.. autosummary::
    :toctree: ../_generated/

    _get_random_initial_experience
    _get_random_types
    _get_random_lagged_choices

likelihood
----------

Routines under ``respy.likelihood`` use the seed specified under
``options["estimation_seed"]`` to control randomness. The seed is used to create the
draws to simulate the probability of observed wages with
:func:`~respy.shared.create_base_draws`.
