Randomness
==========

``respy`` embraces randomness to study individual behavior under risk. At the same time,
it is crucial to hold randomness constant for results to be reproducible. In the
following, all parts of the implementation which rely on randomness are displayed and
documented.

likelihood
----------

Routines under :mod:`respy.likelihood` use the seed specified under
``options["estimation_seed"]`` to control randomness. It is used to create the draws to
simulate the probability of observed wages.

shared
------

The function :func:`respy.shared.create_base_draws` is used in all parts, solution,
simulation, and estimation, to generate random draws.

.. autofunction:: respy.shared.create_base_draws

solve
-----

Routines under :mod:`respy.solve` use the seed specified in the
``options["solution_seed"]`` to control randomness. Apart from the draws,
:func:`respy.solve.solve` relies on the following function.

.. autofunction:: respy.solve.get_not_interpolated_indicator

simulate
--------

Routines under :mod:`respy.simulate` use the seed specified in the
``options["simulation_seed"]`` to control randomness. Apart from the draws,
:func:`respy.simulate.simulate` relies on the following three functions to generate
starting values for simulated agents.

.. autofunction:: respy.simulate._get_random_initial_experience

.. autofunction:: respy.simulate._get_random_types

.. autofunction:: respy.simulate._get_random_lagged_choices
