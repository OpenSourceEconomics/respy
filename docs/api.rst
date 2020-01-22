API
===


User
----

interface
~~~~~~~~~

.. currentmodule:: respy.interface

.. autosummary::
    :toctree: _generated/

    get_example_model
    get_parameter_constraints

solve
~~~~~

.. currentmodule:: respy.solve

.. autosummary::
    :toctree: _generated/

    solve

simulate
~~~~~~~~

.. currentmodule:: respy.simulate

.. autosummary::
    :toctree: _generated/

    get_simulate_func
    simulate

method_of_simulated_moments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: respy.method_of_simulated_moments

.. autosummary::
    :toctree: _generated/

    get_msm_func
    msm

likelihood
~~~~~~~~~~

.. currentmodule:: respy.likelihood

.. autosummary::
    :toctree: _generated/

    get_crit_func
    log_like


Developer
---------

config
~~~~~~

.. currentmodule:: respy.config

.. autosummary::
    :toctree: _generated/

    INDEXER_DTYPE
    INDEXER_INVALID_INDEX

state space
~~~~~~~~~~~

.. currentmodule:: respy.state_space

.. autosummary::
    :toctree: _generated/

    StateSpace
    StateSpace.get_attribute_from_period
    StateSpace.get_continuation_values
    StateSpace.update_systematic_rewards
