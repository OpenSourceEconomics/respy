The State Space
===============

The cartesian product of the core state space and the dense gird maps into the dense
state space. Since this object is growing large and since we want to remain flexible we
split the dense state space in a large number of smaller state spaces. The different
parts and subparts of the state space are explained below:

.. _core_state_space:

.. _dense_grid:

.. _dense_state_space:

.. _period_choice_cores:

.. _dense_period_choice_cores:


The Core State Space
--------------------

.. _state_space_location_indices:


State Space Location Indices
----------------------------

To create the state space and to store information efficiently we build simple indices
of the objects introduced above. In general we call location indices indices if the
defining mapping is injective and keys if the defining mapping is not injective.

.. _core_indices:

Indices are row indices for states in the :ref:`core state space <core_state_space>`.
They are continued over different periods and choice sets in the core.

.. _core_key:

A `core_key` is an index for a set of states in the core state space which are in the
same period and share the same choice set.

.. _dense_vector:

A dense vector is combination of values in the dense dimensions.

.. _dense_index:

A dense index is a position in the dense grid.

.. _dense_key:

A `dense_key` is an index for a set of states in the dense state space which are in the
same period, share the same choice set, and the same dense vector.

.. _complex:

A complex key is the basis for `core_key` and `dense_key` it is a tuple of a period and
a tuple for the choice set which contains booleans for whether a choice is available.
The complex index for a dense index also contains the dense vector in the last position.
