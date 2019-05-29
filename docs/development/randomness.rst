Randomness
==========

``respy`` embraces randomness to study individual behavior under risk. At the same time,
it is crucial to hold randomness constant for results to be reproducible. In the
following, all parts of the implementation which rely on randomness are displayed and
documented.

likelihood
----------

Routines under ``likelihood`` use the seed specified in the ``options_spec`` under
``{"estimation": {"seed"}}`` to control randomness. In ``respy`` it is called
``attr["seed_est"]`` or abbreviated with ``seed`` if no ambiguity exist. It is used to
create the draws to simulate the probability of observed wages.

shared
------

The function :func:`create_multivariate_standard_normal_draws` is used in all parts,
solution, simulation, and estimation, to generate random draws.

.. autofunction:: respy.shared.create_multivariate_standard_normal_draws

solve
-----

Routines under ``solve`` use the seed specified in the ``options_spec`` under
``{"solution": {"seed"}}`` to control randomness. In ``respy`` it is called
``attr["seed_sol"]`` or abbreviated with ``seed`` if no ambiguity exist. Apart from the
draws, :func:`solve` relies on the following function.

.. autofunction:: respy.solve.get_not_interpolated_indicator


simulate
--------

Routines under ``simulate`` use the seed specified in the ``options_spec`` under
``{"simulation": {"seed"}}`` to control randomness. In ``respy`` it is called
``attr["seed_sim"]`` or abbreviated with ``seed`` if no ambiguity exist.Apart from the
draws, :func:`simulate` relies on the following three functions to generate starting
values for simulated agents.

.. autofunction:: respy.simulate._get_random_edu_start

.. autofunction:: respy.simulate._get_random_types

.. autofunction:: respy.simulate._get_random_lagged_choices
