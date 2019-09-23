Randomness and Reproducibility
==============================

``respy`` embraces randomness to study individual behavior under risk. At the same time,
it is crucial to make results reproducible. To build a reproducible model, users must
define three seeds for the solution, simulation and estimation of the model in the
options.

.. code-block:: python

    options = {"solution_seed": 1, "simulation_seed": 2, "estimation_seed": 3}

Each of the seeds starts a sequence of seeds which incrementally increases by one. Thus,
the same seed is never used again and the same randomness is never used twice, but
randomness is still controlled and reproducible. (The need for seed sequences became
apparent in `#268 <https://github.com/OpenSourceEconomics/respy/pull/268>`_.)

As a general rule, models are reproducible with ``respy`` as long as only model
parameters are changed, e.g. utility or type parameters, but the structure of the model
stays the same. The following list includes example of structural changes to the model.

- Changing the choice set (forms of renaming, removing choices).
- Changing the initial conditions (experiences, lagged choices).
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
    :toctree: ../generated/

    simulate_truncated_data
