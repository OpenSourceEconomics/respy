Randomness
==========

``respy`` embraces randomness to study individual behavior under risk. At the same time,
it is crucial to hold randomness constant to make results reproducible. In the
following, all parts of the implementation which rely on randomness are displayed and
documented.

shared
------

The function :func:`create_multivariate_standard_normal_draws` is used in all parts,
solution, simulation, and estimation, to generate random draws.

.. autofunction:: respy.shared.create_multivariate_standard_normal_draws

solve
-----

Apart from the draws, :func:`solve` relies on the following function.

.. autofunction:: respy.solve.get_not_interpolated_indicator


simulate
--------

Apart from the draws, :func:`simulate` relies on the following three functions to
generate starting values for simulated agents.

.. autofunction:: respy.simulate._get_random_edu_start

.. autofunction:: respy.simulate._get_random_types

.. autofunction:: respy.simulate._get_random_lagged_choices
