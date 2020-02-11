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

interpolate
~~~~~~~~~~~

.. currentmodule:: respy.interpolate

.. autosummary::
    :toctree: _generated/

    _get_not_interpolated_indicator

state space
~~~~~~~~~~~

.. currentmodule:: respy.state_space

.. autosummary::
    :toctree: _generated/

    _BaseStateSpace
    _SingleDimStateSpace
    _MultiDimStateSpace
