.. _randomness-and-reproducibility:

Randomness and Reproducibility
==============================

``respy`` embraces randomness to study individual behavior under risk. At the same time,
it is crucial to make results reproducible for research. To build a reproducible model,
users must define three seeds for the solution, simulation and estimation of the model
in the options. This allows to study the impact of randomness for each of the components
independently.

.. code-block:: python

    options = {"solution_seed": 1, "simulation_seed": 2, "estimation_seed": 3}

.. warning::

    Do not use the same seed twice.

The seeds for the solution, simulation and estimation are used to draw a 3-, 5- and
7-digit seed sequence [#f1]_. The first 100 seeds in the sequences are reserved for
randomness in the startup of functions like :func:`~respy.simulate.simulate` or
:func:`~respy.likelihood.log_like`, e.g., to create draws from a uniform distribution.
All other seeds are used during the iterations of those functions and reset to the
initial value at the begin of every iteration.

As a general rule, models in ``respy`` are reproducible or use the same randomness as
long as only model parameters are changed, e.g. utility or type shifts, but the
structure of the model stays the same. The following list includes example of structural
changes to the model.

- Changing the choice set (forms of renaming, removing choices).
- Changing the initial conditions (experiences, lagged choices, type probabilities).
- Changing the Monte Carlo integrations (sequence, number of draws).
- Using interpolation and changing the number of non-interpolated states.
- Removing states from the state space via filters.


In the following, we document for each module the functions which use seeds to control
randomness.

respy.shared
------------

.. currentmodule:: respy.shared

The function :func:`create_base_draws` is used in all parts, solution,
simulation, and estimation, to generate random draws.


.. autosummary::
    :toctree: ../_generated/

    create_base_draws

respy.solve
-----------

.. currentmodule:: respy.solve

Routines under ``respy.solve`` use a seed from the sequence initialized by
``options["solution_seed"]`` to control randomness. Apart from the draws,
:func:`~respy.solve.solve` relies on the following function.


.. autosummary::
    :toctree: ../_generated/

    get_not_interpolated_indicator

respy.simulate
--------------

Routines under ``respy.simulate`` use a seed from the sequence of
``options["simulation_seed"]`` to control randomness. Apart from the draws,
:func:`~respy.simulate.simulate` relies on the following three functions to generate
starting values for simulated agents.

.. currentmodule:: respy.simulate

.. autosummary::
    :toctree: ../_generated/

    _get_random_initial_experience
    _get_random_types
    _get_random_lagged_choices

respy.likelihood
----------------

Routines under ``respy.likelihood`` use a seed from the sequence specified under
``options["estimation_seed"]`` to control randomness. The seed is used to create the
draws to simulate the probability of observed wages with
:func:`~respy.shared.create_base_draws`.


respy.tests.random_model
------------------------

The regression tests are run on truncated data set which contains truncated history of
individuals or missing wage information. The truncation process is controlled via a seed
in the sequence initialized by ``options["simulation_seed"]``.

.. currentmodule:: respy.tests.random_model

.. autosummary::
    :toctree: ../_generated/

    simulate_truncated_data


.. rubric:: Footnotes

.. [#f1] The need for seed sequences became apparent in `#268
         <https://github.com/OpenSourceEconomics/respy/pull/268>`_.
